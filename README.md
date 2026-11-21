# Video indexing at speed

For every second of every FineVideo video in a folder of Parquet files, reports
which of five objects (person, car, dog, laptop, cell phone) YOLO11l finds and
how many people are visible.

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

Install, system packages and the preparation command are filled in below once
the pipeline is final.
