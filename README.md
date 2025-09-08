# K-RAG: Kaspersky Products RAG Service

Минимальный, но рабочий RAG-сервис для ответов на вопросы по двум продуктам Касперского:
- **Kaspersky Security Center 15.1**
- **Kaspersky Anti Targeted Attack Platform 7.1**

## Быстрый запуск

```bash
# Клонирование и запуск
git clone <repository>
cd K-RAG

# Сборка и запуск
docker compose build
docker compose up -d

# Дождитесь инициализации Ollama и загрузки моделей
docker compose logs -f ollama

# Выполните полный цикл сбора данных и индексации
docker compose exec api python -m app.cli.bootstrap

# Проверка работы API
curl -G "http://localhost:8000/ask" --data-urlencode "q=Что нового в KSC 15.1?"
```

## Архитектурные решения

### Модели
- **Эмбеддинги**: `nomic-embed-text` (768-dim, Ollama)
- **Генерация**: `llama3.2` (Ollama)
- **Re-ranking**: `BAAI/bge-reranker-base` (HuggingFace)

### Параметры чанкинга
- **Размер чанка**: 400 слов (первая итерация была произведена с 640)
- **Перекрытие**: 80 слов (первая итерация была произведена с перекрытием в 100)
- **Метаданные**: product, version, url, h1/h2, sha256

### FAISS индекс
- **Тип**: IndexHNSWFlat (Inner Product)
- **M**: 32, **efConstruction**: 200, **efSearch**: 64

### Retrieval pipeline
1. **ANN поиск**: FAISS, topk=15
2. **Cross-encoder re-ranking**: топ-3 контекста
3. **Фильтрация по продукту**: мажоритарный продукт в результатах
4. **Генерация**: llama3.2 с системным промптом (первая итерация была с qwen2.5:7b, но от нее отказался из-за большого количества иероглифов в ответах)

## Результаты mini-evaluation

### Финальные метрики
- **Recall@3**: 0.5 (50%)
- **Exact Match**: 0.8 (80%)
- **Всего вопросов**: 10

### Статистика данных
- **Собранных страниц**: 36 (18 KSC + 18 KATA)
- **Чанков в индексе**: 72 (46 KSC + 26 KATA)
- **Размер FAISS индекса**: ~240KB

### Детализация по продуктам
**KSC 15.1:**
- Базовая страница: `https://support.kaspersky.com/KSC/15.1/ru-RU/5022.htm`
- Успешных ответов: 4/5
- Тематики: новые возможности, требования, настройка, отчеты

**KATA 7.1:**
- Базовая страница: `https://support.kaspersky.com/KATA/7.1/ru-RU/246841.htm`
- Успешных ответов: 4/5
- Тематики: изменения, мультитенантность, Sandbox, виртуализация

## Настройка

### Переменные окружения (.env)
```bash
# Ollama
OLLAMA_URL=http://ollama:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=llama3.2

# Поиск и ранжирование
TOPK=15
TOPN_CONTEXT=3
RERANK_MODEL=BAAI/bge-reranker-base

# Чанкинг
CHUNK_SIZE=400
CHUNK_OVERLAP=80

# FAISS HNSW
HNSW_M=32
HNSW_EF_CONSTRUCTION=200
HNSW_EF_SEARCH=64
```

### Дополнительные источники
```bash
EXTRA_SEEDS="KSC|15.1|https://support.kaspersky.com/KSC/15.1/ru-RU/5022.htm,KATA|7.1|https://support.kaspersky.com/KATA/7.1/ru-RU/246841.htm"
```

## API Endpoints

### GET /health
Проверка состояния сервиса

### GET /ask?q=<вопрос>
Основной endpoint для вопросов

**Пример ответа:**
```json
{
  "answer": "В Kaspersky Security Center 15.1 появились следующие новые возможности...",
  "sources": [
    {
      "url": "https://support.kaspersky.com/KSC/15.1/ru-RU/12521.htm",
      "h1": "Что нового",
      "h2": null
    }
  ]
}
```

## Evaluation

```bash
# Запуск mini-eval
docker compose exec api python -m app.eval.run_eval_sequential

# Просмотр результатов
cat data/eval/eval_report.json
```

## Ограничения

1. **Покрытие данных**: Краулинг глубины 1 от базовых страниц - не все страницы Касперского могут быть найдены
2. **Языковые артефакты**: Иногда `llama3.2` может генерировать символы CJK (очищаются автоматически)
3. **Таймауты**: LLM генерация может быть медленной на слабом железе
4. **Точность**: Re-ranking помогает, но качество сильно зависит от полноты краулинга

## Улучшения

### Краткосрочные
- Добавить больше seed URL для расширения покрытия
- Настроить более агрессивный краулинг (глубина 2+)
- Оптимизировать промпты для лучшей генерации

### Долгосрочные
- Интегрировать sitemap.xml для полного покрытия
- Добавить переиндексацию в реальном времени
- Реализовать фильтрацию по датам и версиям
- Добавить поддержку мультиязычности

## Отладка

```bash
# Логи Ollama
docker compose logs ollama

# Логи API
docker compose logs api

# Проверка индекса
docker compose exec api python -c "from app.index.faiss_store import load_index; i,m = load_index(); print(f'Index: {i.ntotal} vectors')"

# Тест поиска
docker compose exec api python -c "from app.pipeline import get_pipeline; p = get_pipeline(); hits,_ = p.retriever.ann_search('установка KSC', 5); [print(f'{h[\"_sim\"]:.3f}: {h[\"url\"]}') for h in hits]"
```

---
*Разработано как минимальная, но рабочая демонстрация RAG-системы для документации Касперского*