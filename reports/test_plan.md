# Test plan (frozen before the test split is opened)

Everything below was decided on the train and validation splits only. The test split is evaluated once, with these
models, thresholds and analyses. Nothing is changed after test results are seen; any analysis not listed here is
reported as exploratory.

## Pre-test checks (validation, all passed)

- **Labels.** For all 450 SAD-Syn validation pages, the labels served by the training dataset match the manifest, and
  the manifest matches the generated document (rebuilt with `plan_documents`). The true values of every altered page
  break its expected rule; no clean page breaks any rule (`doc_verify.checks.ground_truth_check`).
- **Evidence is in the text the model sees.** Where OCR read every evidence value (83.6% of pages), all of it lies
  inside the 8 text windows (latest evidence ends at token 381 of 425).
- **Text cache.** On Colab, every one of the 6,482 cached token masks equals the mask recomputed from its page's text;
  recomputed CLIP features differ from their own cached row by at most 0.047 and from the next page's row by at
  least 12 (`doc_verify.checks.cache_check`).
- **Saved predictions.** Every run's validation predictions carry the labels of the page at their `page_index`.

## Frozen models

| Model | Role | Checkpoint (best validation epoch) |
|---|---|---|
| Fused ViT-B/16 + CLIP text, λ₁ = λ₂ = 1 | main model | `runs/fused/best.pt` (epoch 8) |
| Fused, λ₂ = 0 | ablation of the kind head | `runs/fused_no_kind/best.pt` (epoch 5) |
| Image-only ViT-B/16 | image baseline | `runs/image_vit/best.pt` (epoch 10) |
| ResNet18 from scratch | type baseline | `runs/resnet18/best.pt` (epoch 7) |
| TF-IDF + logistic regression | text baseline | refit on train exactly as for validation |
| CLIP ViT-B/32 zero-shot, prompt "a scanned {class}" | type baseline | no training |
| Rule checker on OCR values | authenticity baseline | no training |

The two authenticity-only runs (`fused_auth_only`, `fused_auth_long`) are diagnostics and are reported on validation
only.

## Decision thresholds (chosen on validation: best balanced accuracy)

A page is called altered when its score is at or above the threshold (`reports/val_thresholds.json`).

| Model | Threshold |
|---|---|
| Fused | 0.3989 |
| Fused, λ₂ = 0 | 0.4847 |
| Image-only ViT | 0.6996 |
| TF-IDF + LR | 0.3937 |
| Rule checker | 1 failed rule |

## Metrics and planned analyses

1. **Document type** (Tobacco3482 test, 521 pages): accuracy and macro-F1 with 95% bootstrap intervals (2,000
   resamples of pages, seed 0); a per-class confusion matrix for every type model.
2. **Authenticity** (SAD-Syn test, 450 pages, 225 pairs): AUC with 95% intervals from 2,000 resamples of clean/altered
   pairs (seed 0); accuracy and macro-F1 at the thresholds above.
3. **Authenticity breakdowns**, each over both partners of the pairs in the group, with pair-bootstrap intervals: by
   inconsistency kind, by document type, and by whether OCR read the altered value (`altered_value_ocr_ok`).
4. **Kind** macro-F1 (five classes) for the models with a kind head and the rule checker.
5. **McNemar's test** (exact) on per-page right/wrong decisions, with Holm correction across the three tests:
   - type: fused vs image-only ViT (the stronger single-modality model on validation);
   - authenticity: fused vs rule checker, at the thresholds above;
   - type: fused vs fused λ₂ = 0 (the kind-head ablation).
   Clean and altered partners share a template and scan noise, so the authenticity test treats pages as independent
   when they are not fully; this is stated alongside the result.

## Validation findings that set expectations

- Type macro-F1: fused 0.849, fused λ₂ = 0 0.859, image-only ViT 0.818, TF-IDF 0.706, ResNet18 0.544, CLIP zero-shot
  0.433.
- Authenticity AUC: rule checker 0.758 (0.72–0.80); every learned model is at chance (fused 0.525, 0.49–0.56; image-only
  0.527; TF-IDF 0.485). The 40-epoch authenticity-only run fits the training pages (loss 0.69 → 0.20) while validation
  AUC stays at 0.50, so the learned models memorise rather than learn the cross-field comparison.
- By kind, the fused model is at chance on every kind, string comparisons (institution mismatch, 0.54) as well as
  numbers. Across 50 subgroup intervals for the five neural runs, 3 barely exclude 0.5, all in the diagnostic runs;
  about 2–3 are expected by chance, so these are not read as signal.
- The rule checker is weakest on totals that do not reconcile (0.65) and on pages where OCR lost the altered value
  (0.61).
