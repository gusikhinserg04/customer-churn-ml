
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)


DATA_PATH = Path("data/Telco-Customer-Churn — копия.csv")
MODEL_PATH = Path("models/churn_model.joblib")


def main():
    print(f"pandas version: {pd.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")

    # 1. Загружаем данные
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print("\nИсходный размер датасета:", df.shape)

    # 2. Проверяем необходимые столбцы
    required_columns = {"customerID", "TotalCharges", "Churn"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print("В датасете отсутствуют столбцы:", sorted(missing_columns))
        return

    # 3. Преобразуем TotalCharges в числовой формат
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype("string").str.strip(),
        errors="coerce",
    )

    # 4. Удаляем полные дубликаты
    duplicates = df.duplicated().sum()
    print("Количество полных дубликатов:", duplicates)

    df = df.drop_duplicates().copy()
    print("Размер после удаления дубликатов:", df.shape)

    # 5. Заполняем пропуски в целевой переменной и проверяем её
    df = df.dropna(subset=["Churn"])
    df = df[df["Churn"].isin(["Yes", "No"])].copy()

    # Удаляем идентификатор клиента
    df = df.drop(columns=["customerID"])

    # Преобразуем целевую переменную:
    # 1 — клиент ушёл, 0 — остался
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # 6. Создаём признаки
    X = df.drop(columns=["Churn"])

    print("\nРазмер X:", X.shape)
    print("Размер y:", y.shape)
    print("\nРаспределение целевой переменной:")
    print(y.value_counts())

    # 7. Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nРазмер обучающей выборки:", X_train.shape)
    print("Размер тестовой выборки:", X_test.shape)

    # 8. Определяем типы признаков
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print("\nЧисловые признаки:", numeric_features)
    print("Категориальные признаки:", categorical_features)

    # 9. Подготавливаем числовые признаки
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 10. Кодируем категориальные признаки
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 11. Объединяем обработку признаков
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # 12. Создаём единый конвейер обработки и обучения
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # 13. Обучаем модель
    print("\nОбучаем модель...")
    model.fit(X_train, y_train)
    print("Модель успешно обучена!")

    # 14. Получаем прогнозы
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 15. Оцениваем качество
    print("\n--- РЕЗУЛЬТАТЫ МОДЕЛИ ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.3f}")

    print("\nОтчёт классификации:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Остался", "Ушёл"],
            zero_division=0,
        )
    )

    print("Матрица ошибок:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # 16. Сохраняем модель
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nМодель сохранена: {MODEL_PATH}")

    print("\nПроект успешно выполнен!")


if __name__ == "__main__":
    main()

