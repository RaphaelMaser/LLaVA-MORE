import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


def _parse_aloe_reference(reference: str) -> tuple[str, str]:
	if not reference.startswith("aloe://"):
		raise ValueError(f"Invalid aloe vision tower reference: {reference}")

	payload = reference[len("aloe://"):].strip("/")
	if not payload:
		raise ValueError("Aloe vision tower reference must include at least a backbone name")

	parts = payload.split("/")
	if len(parts) == 1:
		return "distilled", parts[0]
	if len(parts) == 2:
		return parts[0], parts[1]

	raise ValueError(
		"Aloe vision tower reference must have the form aloe://<backbone> or aloe://<model_type>/<backbone>"
	)


def _resolve_aloe_device(device_map) -> str:
	if isinstance(device_map, dict) and device_map:
		first_device = next(iter(device_map.values()))
		return str(first_device)
	if isinstance(device_map, str) and device_map != "auto":
		return device_map
	return "cuda" if torch.cuda.is_available() else "cpu"


class AloeVisionTower(nn.Module):
	def __init__(self, vision_tower, args, delay_load=False):
		super().__init__()

		self.is_loaded = False
		self.vision_tower_name = vision_tower
		self.model_type, self.backbone_name = _parse_aloe_reference(vision_tower)
		self.select_layer = args.mm_vision_select_layer
		self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
		self.repo_root = os.environ.get("ALOE_REPO_ROOT")
		if not self.repo_root:
			raise ValueError("ALOE_REPO_ROOT must be set to use aloe vision towers")

		self.bridge_module_path = Path(self.repo_root) / "aloe_model_loader.py"
		if not self.bridge_module_path.exists():
			raise ValueError(f"Could not find aloe bridge module at {self.bridge_module_path}")

		self._bridge_module = None
		self.target_model_config = None
		self.cfg_only = None

		if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
			self.load_model()
		else:
			self._initialize_cfg_only()

	def _load_bridge_module(self):
		if self._bridge_module is not None:
			return self._bridge_module

		spec = importlib.util.spec_from_file_location("aloe_model_loader_bridge", self.bridge_module_path)
		if spec is None or spec.loader is None:
			raise ImportError(f"Failed to load aloe bridge module from {self.bridge_module_path}")

		module = importlib.util.module_from_spec(spec)
		sys.modules[spec.name] = module
		spec.loader.exec_module(module)
		self._bridge_module = module
		return module

	@staticmethod
	def _image_size_from_processor_size(size_config) -> int:
		if isinstance(size_config, dict):
			if "height" in size_config:
				return int(size_config["height"])
			if "shortest_edge" in size_config:
				return int(size_config["shortest_edge"])
		if isinstance(size_config, int):
			return size_config
		raise ValueError(f"Unsupported processor size config: {size_config}")

	def _initialize_cfg_only(self):
		bridge_module = self._load_bridge_module()
		cfg = bridge_module.compose_aloe_model_config(
			backbone_name=self.backbone_name,
			model_type=self.model_type,
		)
		self.target_model_config = cfg.model_factory.target_model_config
		override_res = self.target_model_config.get("override_resolution", None)
		image_processor = bridge_module.load_backbone_image_processor(self.target_model_config.name, override_res)
		self.cfg_only = SimpleNamespace(
			hidden_size=self.target_model_config.feature_dim,
			image_size=self._image_size_from_processor_size(image_processor.size),
			patch_size=self.target_model_config.patch_size,
		)

	def load_model(self, device_map=None, torch_dtype=None):
		if self.is_loaded:
			print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
			return

		print(
			f"Loading ALOE vision tower {self.vision_tower_name} "
			f"(model_type={self.model_type}, backbone={self.backbone_name})"
		)

		bridge_module = self._load_bridge_module()
		bundle = bridge_module.get_aloe_model_bundle(
			backbone_name=self.backbone_name,
			model_type=self.model_type,
			device=_resolve_aloe_device(device_map),
			compile_model=False,
			use_gradient_checkpointing=False,
		)

		self.vision_tower = bundle.model
		if torch_dtype is not None:
			self.vision_tower = self.vision_tower.to(dtype=torch_dtype)
		self.image_processor = bundle.image_processor
		self.target_model_config = bundle.model_factory_config.target_model_config
		self.cfg_only = SimpleNamespace(
			hidden_size=self.target_model_config.feature_dim,
			image_size=self._image_size_from_processor_size(self.image_processor.size),
			patch_size=self.target_model_config.patch_size,
		)

		self.vision_tower.requires_grad_(False)
		self.is_loaded = True

	def feature_select(self, image_forward_outs):
		image_features = image_forward_outs.hidden_states[self.select_layer]
		if self.select_feature == 'patch':
			# Match SigLIP2VisionTower behavior: keep all tokens for 'patch'.
			# (Some backbones may have a CLS token; we intentionally do not strip it here.)
			image_features = image_features[:, :]
		elif self.select_feature == 'cls_patch':
			image_features = image_features
		else:
			raise ValueError(f'Unexpected select feature: {self.select_feature}')
		return image_features

	@torch.no_grad()
	def forward(self, images):
		if type(images) is list:
			image_features = []
			for image in images:
				image_forward_out = self.vision_tower(
					image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
					output_hidden_states=True,
				)
				image_feature = self.feature_select(image_forward_out).to(image.dtype)
				image_features.append(image_feature)
		else:
			image_forward_outs = self.vision_tower(
				images.to(device=self.device, dtype=self.dtype),
				output_hidden_states=True,
			)
			image_features = self.feature_select(image_forward_outs).to(images.dtype)

		return image_features

	@property
	def dummy_feature(self):
		return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

	@property
	def dtype(self):
		p = next(self.vision_tower.parameters(), None)
		return p.dtype if p is not None else torch.float32

	@property
	def device(self):
		p = next(self.vision_tower.parameters(), None)
		return p.device if p is not None else torch.device('cpu')

	@property
	def config(self):
		return self.cfg_only

	@property
	def hidden_size(self):
		return self.cfg_only.hidden_size

	@property
	def num_patches_per_side(self):
		return self.cfg_only.image_size // self.cfg_only.patch_size

	@property
	def num_patches(self):
		return self.num_patches_per_side ** 2
