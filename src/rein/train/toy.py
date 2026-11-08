from rein.schema import Message, Question, Split, State, TrainingRecord

PHOTO_TEXT = "*leans on the counter* send me a photo please"
CHAT_TEXT = "*sips the coffee* tell me about your day"


def toy_records(split: Split, count: int, filler_turns: int = 0) -> list[TrainingRecord]:
    """Small learnable records (photo request yes/no + tool choice) for tests and Pod smoke runs."""
    records = []
    for i in range(count):
        wants_photo = i % 2 == 0
        filler = tuple(
            Message(role="user" if t % 2 == 0 else "assistant", text=f"*small talk {t}* the weather is nice today")
            for t in range(filler_turns)
        )
        latest = Message(role="user", text=PHOTO_TEXT if wants_photo else CHAT_TEXT)
        records.append(
            TrainingRecord(
                id=f"{split}-{i}",
                source="toy",
                scenario_id=f"toy-{split}-{i}",
                split=split,
                state=State(context="Luna is a barista.", conversation=(*filler, latest)),
                questions={
                    "photo": Question(type="noul", instructions="Did the user ask for a photo?", label=wants_photo),
                    "tool": Question(
                        type="choice",
                        instructions="Which tool does the latest message need?",
                        criteria={"send_photo": "share an image", "web_search": None, "none": None},
                        label="send_photo" if wants_photo else "none",
                    ),
                },
            )
        )
    return records
