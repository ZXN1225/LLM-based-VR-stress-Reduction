from pydantic import BaseModel, ConfigDict, Field, model_validator


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int = Field(ge=1, le=10)
    duration: int = Field(gt=0, le=600)
    image_prompt: str = Field(min_length=3)
    target_kelvin: float = Field(ge=1500, le=12000)
    intensity: float = Field(gt=0, le=10)
    therapeutic_goal: str = Field(min_length=1)


class MusicSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int = Field(ge=1)
    music_prompt: str = Field(min_length=3)
    style: str = Field(min_length=1)
    title: str = Field(min_length=1)


class InterventionPlanSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    scenes: list[SceneSpec] = Field(min_length=10, max_length=10)
    music_playlist: list[MusicSpec] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_steps(self):
        steps = [scene.step for scene in self.scenes]
        if len(set(steps)) != len(steps) or set(steps) != set(range(1, 11)):
            raise ValueError("scene steps must be unique integers 1 through 10")
        return self


class RefinementPlanSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    scenes: list[SceneSpec] = Field(min_length=1, max_length=1)
