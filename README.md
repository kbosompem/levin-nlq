# levin-nlq

Natural Language to Datalevin Query model training.

Converts English queries to Datalog syntax specifically for **Datalevin** - not Datomic or DataScript.

## Goal

A ~70MB quantized model that runs offline, converting:
```
"Find all users who joined after 2024 with gmail addresses"
```
to:
```clojure
[:find ?e ?name ?email
 :where
 [?e :user/name ?name]
 [?e :user/email ?email]
 [?e :user/joined ?date]
 [(> ?date #inst "2024-01-01")]
 [(clojure.string/includes? ?email "gmail")]]
```

## Structure

```
levin-nlq/
├── docs/
│   └── datalevin-differences.md   # Critical: what makes Datalevin unique
├── training-data/
│   ├── train.jsonl                # Training examples
│   └── valid.jsonl                # Validation examples
├── scripts/
│   ├── generate-data.py           # Training data generator
│   ├── train.sh                   # Fine-tuning script
│   └── export-onnx.sh             # ONNX export script
├── models/                        # Output models (gitignored except releases)
└── experiments/                   # Training experiments (gitignored)
```

## Datalevin vs Others

See [docs/datalevin-differences.md](docs/datalevin-differences.md) for critical differences from Datomic/DataScript that affect query generation.

## Usage

### 1. Generate Training Data
```bash
python scripts/generate-data.py
```

### 2. Fine-tune (Mac Mini M1)
```bash
./scripts/train.sh
```

### 3. Export to ONNX
```bash
./scripts/export-onnx.sh
```

## Target Model

- Base: SmolLM-135M-Instruct
- Fine-tuning: LoRA
- Quantization: 4-bit
- Format: ONNX
- Size: ~70MB

## Consumer

Primary consumer is the [Levin VS Code extension](https://github.com/your-org/levin), but the model could be used in other contexts (CLI tools, web interfaces, etc.).
