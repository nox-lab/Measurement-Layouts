from animalai.environment import AnimalAIEnvironment
class AnimalAIReset(AnimalAIEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes = kwargs
        for key in kwargs:
            setattr(self, key, kwargs[key])
    def reset_arenas(self):
        super().close()
        super().__init__(**self.attributes)
        return self
        