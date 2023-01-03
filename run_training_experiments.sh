# Establish the benchmark
poetry run python src/benchmark.py --experiment='benchmark'

# Run with all features
poetry run python src/train.py \
    --n-estimators=150 \
    --learning-rate=0.5 \
    --colsample-bytree=1.0 \
    --max-depth=20 \
    --subsample=1.0 \
    --experiment='all_features_xgb'

# Run with filtered features
poetry run python src/train.py \
    --n-estimators=150 \
    --learning-rate=0.5 \
    --colsample-bytree=1.0 \
    --max-depth=20 \
    --subsample=1.0 \
    --experiment='only_important_features_xgb' \
    --final-features

