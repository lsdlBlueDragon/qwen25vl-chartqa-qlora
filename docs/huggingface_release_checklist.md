# Hugging Face release checklist

## Adapter repository

- [ ] Choose the public adapter repository ID.
- [ ] Copy the hardmix adapter files from the private archive.
- [ ] Verify `adapter_config.json` points to `Qwen/Qwen2.5-VL-3B-Instruct`.
- [ ] Calculate and record the SHA-256 of `adapter_model.safetensors`.
- [ ] Replace `<HUGGING_FACE_ADAPTER_REPO>` in `deployment/adapter/README.md`.
- [ ] Add the project repository URL and exact training configuration.
- [ ] Upload only adapter, tokenizer/processor metadata when required, and the model card.
- [ ] Test loading the published revision on a clean GPU runtime.

## Space repository

- [ ] Choose the Space repository ID and GPU hardware.
- [ ] Publish the project code or a clean deployment branch.
- [ ] Use `deployment/space/README.md.template` as the Space root `README.md`.
- [ ] Set `CHARTQA_ADAPTER_PATH` to the published adapter repository ID.
- [ ] Confirm the Space uses the deployment requirements.
- [ ] Verify cold start, built-in example loading, base inference, adapter inference, and model switching.
- [ ] Record a screenshot and public URL in the project README.

## Release gate

Do not publish until placeholders are replaced, credentials are configured outside Git, and the hardmix adapter checksum matches the private G drive artifact.
