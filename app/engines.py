"""Shared references to the loaded models, populated once at app startup."""


class Engines:
    def __init__(self):
        self.voice = None  # app.voice.pipeline.Transcriber
        self.face = None   # app.face.faceapi.FaceEngine
        self.analyzer = None  # app.voice.analyze.ConversationAnalyzer (local LLM)


engines = Engines()
