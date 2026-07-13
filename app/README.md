# Gradio App

The entry point is `app/app.py`. Build the interface without loading the model:

```powershell
python app\app.py --dry-run
```

Launch the server on a GPU environment:

```powershell
python app\app.py --host 0.0.0.0 --port 7860
```

The app supports:

- image upload,
- chart question input,
- base/adapted model selector,
- answer output,
- latency display,
- three built-in questions using the self-created `examples/quarterly_sales.svg` chart.

Configuration is provided through environment variables:

- `CHARTQA_MODEL_ID`, default `Qwen/Qwen2.5-VL-3B-Instruct`;
- `CHARTQA_ADAPTER_PATH`, default `outputs/adapters/chartqa_qlora_hardmix1k_steps100`;
- `CHARTQA_LOAD_IN_4BIT`, default `1`;
- `CHARTQA_MIN_PIXELS` and `CHARTQA_MAX_PIXELS`.

Only one model configuration is kept in memory. Switching between base and adapter mode releases the previous model before loading the next one.
