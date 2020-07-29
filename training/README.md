# Training :computer: :fire:

1. **Fork `Training-demo` notebook**

Naming convention: E.g. `initials-yyyy-MM-dd-runningIndexPerDay` but this is not forced. Description is also fine. 

2. **Change magic variables :wrench:**

The notebook follows a template but that shouldn't restrict your experiments. If you wan't to use a new model, check how it's done in `model/___init__.py`.

If you wan't to tune transforms, epochs, optimizers, dropout or learning rates, these are all in the **variables** cell. Additionally, the training cell is still exposed so you can work your magic there if one_cycle schedule isn't your thing.

3. **Remember to use a unique SAVE_DIR and check data statistics (mean and std)**

## Results

The trained models are saved to your SAVE_DIR along with training metadata and evaluation results.

## Kaggle Inference

work in progress...

### This is cool but I need to...

If you have an idea that's not possible to implement with these scripts, please create a new dev branch and customize it to your needs. You can later make a pull request.

-------------------------------------------

## Evaluating a model directory without a config

If you have a set of CV models in a directory without a config file, you can still use the evaluation script by providing some variables manually.

In training directory, run:

```bash
python ./evaluation/evaluate_folder.py --model_dir "../concat_tile_pooling/models/tile_128_2020_05_03/" --train_image_dir "../../tile_data/train/" --train_csv "../../prostate-cancer-grade-assessment/train.csv" --mean 0.0905 0.1811 0.1220 --std 0.3635, 0.4998, 0.4047 --arch "resnext50_32x4d_ssl"
```

This will create cv-fold specific confusion matrices and scores.json to your model dir.