from typing import Dict

model_names = ['rtdetr-x.pt', 'rtdetr-l.pt', 'deyo-x.pt', 'yolov8x.pt', 'yolov9c.pt', 'yolov9e.pt', 'gelan-e.pt']
model_name = model_names[0]


class DetectionModel:
    _model = model_names[0]

    @classmethod
    def set_model(cls, model_name) -> Dict[str, str]:
        if model_name not in model_names:
            raise ValueError("Invalid model name")
        cls._model = model_name
        return {"model": cls._model}

    @classmethod
    def get_model(cls) -> Dict[str, str]:
        return {"model": cls._model}

    @classmethod
    def get_models(cls) -> Dict[str, list]:
        return {"models": model_names}
