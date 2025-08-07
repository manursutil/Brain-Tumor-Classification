import io
import types

import pytest
import torch
from PIL import Image

import src.evaluate as ev


class DummyModel(torch.nn.Module):
    """Tiny model with .device and .fc like resnet18, but fully stubbed."""
    def __init__(self, logits_fn=None, in_features=512):
        super().__init__()
        # mimic resnet18 having an fc with in_features
        self.fc = types.SimpleNamespace(in_features=in_features)
        self._eval_called = False
        self.device = torch.device("cpu")
        # function that returns logits for given input batch
        self._logits_fn = logits_fn or (lambda x: torch.zeros((x.shape[0], 4)))

    def forward(self, x):
        return self._logits_fn(x)

    def eval(self):
        self._eval_called = True
        return super().eval()

    def load_state_dict(self, *_args, **_kwargs):
        # pretend weights loaded fine
        return


def tiny_transform(_img):
    # mimic torchvision transform -> tensor (3,224,224)
    return torch.zeros(3, 224, 224)


def make_jpeg_bytes(size=(32, 32), color=(120, 50, 200)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def patch_class_names(monkeypatch):
    # Ensure stable class names
    monkeypatch.setattr(ev, "CLASS_NAMES", ["glioma", "meningioma", "notumor", "pituitary"], raising=True)


def test_load_model_initializes_and_sets_eval_and_device(monkeypatch):
    dummy = DummyModel()
    # Patch resnet18 to return our dummy
    monkeypatch.setattr(ev.models, "resnet18", lambda weights=None: dummy, raising=True)
    # Patch torch.load to avoid filesystem
    monkeypatch.setattr(ev.torch, "load", lambda *a, **k: {}, raising=True)

    model = ev.load_model(weights_path="irrelevant.pt")

    assert model is dummy
    assert dummy._eval_called is True
    # attribute is set by load_model
    assert hasattr(model, "device")
    # should be cpu or cuda depending on environment, but device exists
    assert isinstance(model.device, torch.device)


def test_predict_image_with_pil_image(monkeypatch):
    # Logits so class index 2 is top (notumor)
    def logits(x):
        batch = x.shape[0]
        out = torch.zeros(batch, 4)
        out[:, 2] = 5.0
        return out

    model = DummyModel(logits_fn=logits)
    model.device = torch.device("cpu")

    # patch transforms
    monkeypatch.setattr(ev, "get_transforms", lambda: tiny_transform, raising=True)

    img = Image.new("RGB", (64, 64), (10, 200, 30))
    result = ev.predict_image(model, img)

    assert result["class_index"] == 2
    assert result["predicted_class"] == ev.CLASS_NAMES[2]
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_image_with_path_string(monkeypatch, tmp_path):
    # Make a real JPEG file
    img_bytes = make_jpeg_bytes()
    p = tmp_path / "x.jpg"
    p.write_bytes(img_bytes)

    # Force class 1
    def logits(x):
        out = torch.zeros(x.shape[0], 4)
        out[:, 1] = 3.14
        return out

    model = DummyModel(logits_fn=logits)
    model.device = torch.device("cpu")

    monkeypatch.setattr(ev, "get_transforms", lambda: tiny_transform, raising=True)

    result = ev.predict_image(model, str(p))
    assert result["class_index"] == 1
    assert result["predicted_class"] == ev.CLASS_NAMES[1]


def test_evaluate_model_happy_path(monkeypatch):
    """
    Build a one-batch DataLoader with labels [0,1,2,3] and produce matching predictions,
    so accuracy is 1.0 and confusion matrix is identity 4x4.
    """
    # dataset -> (images, labels)
    images = torch.zeros(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])

    class FakeDataset:
        def __len__(self): return 4
        def __getitem__(self, idx):  # not used because we replace DataLoader
            return images[idx], labels[idx]

    # Patch dataset constructor & transforms
    monkeypatch.setattr(ev, "BrainTumorDataset", lambda *a, **k: FakeDataset(), raising=True)
    monkeypatch.setattr(ev, "get_transforms", lambda: tiny_transform, raising=True)

    # Patch DataLoader to yield our single batch directly
    def fake_loader(_ds, batch_size=32):
        # return an object that is iterable once, yielding (images, labels)
        return [(images, labels)]
    monkeypatch.setattr(ev, "DataLoader", fake_loader, raising=True)

    # Model that predicts exactly the true labels
    def logits(x):
        # Make argmax == [0,1,2,3]
        out = torch.zeros(x.shape[0], 4)
        for i in range(x.shape[0]):
            out[i, i] = 10.0
        return out

    model = DummyModel(logits_fn=logits)
    model.device = torch.device("cpu")

    result = ev.evaluate_model(model, test_dir="ignored", batch_size=4)

    assert "accuracy" in result and result["accuracy"] == pytest.approx(1.0, rel=1e-6)
    assert "classification_report" in result and isinstance(result["classification_report"], dict)
    assert "confusion_matrix" in result
    # 4x4 identity
    assert result["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_evaluate_model_handles_nonperfect_preds(monkeypatch):
    """
    Make half the predictions wrong; just validate keys are present and accuracy in (0,1).
    """
    images = torch.zeros(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])

    class FakeDataset:
        def __len__(self): return 4

    monkeypatch.setattr(ev, "BrainTumorDataset", lambda *a, **k: FakeDataset(), raising=True)
    monkeypatch.setattr(ev, "get_transforms", lambda: tiny_transform, raising=True)

    def fake_loader(_ds, batch_size=32):
        return [(images, labels)]

    monkeypatch.setattr(ev, "DataLoader", fake_loader, raising=True)

    # Predict [1,1,2,2] -> accuracy 0.5
    def logits(x):
        out = torch.zeros(x.shape[0], 4)
        out[0, 1] = 1.0
        out[1, 1] = 1.0
        out[2, 2] = 1.0
        out[3, 2] = 1.0
        return out

    model = DummyModel(logits_fn=logits)
    model.device = torch.device("cpu")

    result = ev.evaluate_model(model, test_dir="ignored", batch_size=4)

    assert 0.0 < result["accuracy"] < 1.0
    assert isinstance(result["classification_report"], dict)
    assert isinstance(result["confusion_matrix"], list)
