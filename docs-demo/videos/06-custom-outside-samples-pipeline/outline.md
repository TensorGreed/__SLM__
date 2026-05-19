# Video 06: Custom Outside-Samples Pipeline Outline

Status: conceptual/partial. This is not an official sample template.

Possible custom flow:
1. Start with a raw dataset.
2. Clean it.
3. Create or import gold examples.
4. Generate synthetic examples.
5. Prepare a training dataset.
6. Configure training.
7. Evaluate.
8. Compress/export.
9. Use the trained model.

Evidence constraints:
- Bring-your-own dataset is partially supported by ingestion/import UI and APIs.
- Cleaning UI/API exists.
- Gold create/import surfaces exist.
- Synthetic generation exists but may need teacher model or demo fallback.
- Training, compression, export, and final usage are runtime-dependent.

Do not claim the repo supports every step until a real run proves it.

