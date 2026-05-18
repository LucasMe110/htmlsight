import json
from dataclasses import dataclass
from pathlib import Path

DISPLAY_VALUES = ["block", "inline", "inline-block", "flex", "grid", "none", "other"]
ROLE_VALUES = ["button", "link", "alert", "navigation", "tab", "dialog", "none", "other"]


@dataclass(frozen=True)
class AttributeVocab:
    tag_to_id: dict[str, int]
    display_to_id: dict[str, int]
    role_to_id: dict[str, int]

    @classmethod
    def from_observations(
        cls,
        tags: list[str],
        displays: list[str],
        roles: list[str | None],
    ) -> "AttributeVocab":
        tag_values = sorted(set(tags) | {"other"})
        return cls(
            tag_to_id={value: index for index, value in enumerate(tag_values)},
            display_to_id={value: index for index, value in enumerate(DISPLAY_VALUES)},
            role_to_id={value: index for index, value in enumerate(ROLE_VALUES)},
        )

    def encode_tag(self, value: str) -> int:
        return self.tag_to_id.get(value, self.tag_to_id["other"])

    def encode_display(self, value: str) -> int:
        return self.display_to_id.get(value, self.display_to_id["other"])

    def encode_role(self, value: str | None) -> int:
        normalized = "none" if value is None else value
        return self.role_to_id.get(normalized, self.role_to_id["other"])

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "tag_to_id": self.tag_to_id,
                    "display_to_id": self.display_to_id,
                    "role_to_id": self.role_to_id,
                },
                indent=2,
                sort_keys=True,
            )
        )

    @classmethod
    def load(cls, path: Path) -> "AttributeVocab":
        payload = json.loads(path.read_text())
        return cls(
            tag_to_id=payload["tag_to_id"],
            display_to_id=payload["display_to_id"],
            role_to_id=payload["role_to_id"],
        )
