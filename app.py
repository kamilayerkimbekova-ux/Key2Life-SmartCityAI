import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import gradio as gr

texts = [
    "На дороге большая яма",
    "Не работает фонарь возле дома",
    "Переполнены мусорные баки",
    "Опасная драка во дворе",
    "Автобус постоянно опаздывает",
    "Разбит асфальт на улице",
    "Нет освещения на перекрестке",
    "Мусор не вывозят уже неделю",
    "Подозрительные люди возле школы",
    "Маршрутка не приехала",
    "Огромная яма возле остановки",
    "Сломан уличный фонарь",
    "Свалка мусора возле дома",
    "Небезопасный участок дороги",
    "Проблемы с общественным транспортом"
]

labels = [
    "Дороги",
    "Освещение",
    "Мусор",
    "Безопасность",
    "Транспорт",
    "Дороги",
    "Освещение",
    "Мусор",
    "Безопасность",
    "Транспорт",
    "Дороги",
    "Освещение",
    "Мусор",
    "Безопасность",
    "Транспорт"
]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(texts, labels)

def analyze(text):
    category = model.predict([text])[0]

    high_words = ["авария", "опасно", "срочно", "драка", "угроза"]

    priority = "Высокий"
    if not any(word in text.lower() for word in high_words):
        priority = "Средний"

    return f"Категория: {category}\nПриоритет: {priority}"

demo = gr.Interface(
    fn=analyze,
    inputs="text",
    outputs="text",
    title="SmartCity AI Assistant",
    description="Система автоматической обработки обращений граждан"
)

demo.launch()
