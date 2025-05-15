# Установка необходимых пакетов (если они не установлены)
# install.packages(c("tidyverse", "randomForest", "caret", "ggplot2", "fastDummies", "reshape2"))

# Подключение библиотек
library(tidyverse)
library(randomForest)
library(caret)
library(ggplot2)
library(fastDummies)
library(reshape2)

# Загрузка данных
df <- read.csv("credit_risk_dataset.csv")

# Проверка данных
cat("Первые 5 строк данных:\n")
print(head(df, 5))

cat("\nСтруктура данных:\n")
print(str(df))

cat("\nПропущенные значения:\n")
print(colSums(is.na(df)))

# Обработка пропущенных значений
df <- df %>%
  mutate(
    loan_int_rate = ifelse(is.na(loan_int_rate),
      mean(loan_int_rate, na.rm = TRUE),
      loan_int_rate
    ),
    person_emp_length = ifelse(is.na(person_emp_length),
      mean(person_emp_length, na.rm = TRUE),
      person_emp_length
    )
  )

cat("\nПропущенные значения:\n")
print(colSums(is.na(df)))

# Логарифмирование для обработки выбросов
df <- df %>% mutate(person_income = log10(person_income))


# Фильтрация целевого признака
df <- df %>% filter(loan_status %in% c(0, 1))
df$loan_status <- factor(df$loan_status, levels = c(0, 1))

# Выбираем ключевые числовые переменные
num_vars <- c(
  "person_age", "person_income", "person_emp_length",
  "loan_amnt", "loan_int_rate", "loan_percent_income"
)

for (var in num_vars) {
  if (var %in% names(df)) {
    p <- ggplot(df, aes(x = .data[[var]])) +
      geom_histogram(bins = 30, fill = "steelblue", alpha = 0.8, color = "white") +
      labs(
        title = paste("Распределение", var),
        subtitle = ifelse(var %in% c("person_emp_length", "loan_int_rate"),
          "После обработки пропусков", ""
        ),
        x = var, y = "Частота"
      ) +
      theme_minimal()
    print(p)
    readline(prompt = "Нажмите Enter для продолжения")
  }
}



cat_vars <- c(
  "person_home_ownership", "loan_intent", "loan_grade",
  "cb_person_default_on_file"
)

for (cat_var in cat_vars) {
  if (cat_var %in% names(df)) {
    for (num_var in c("loan_int_rate", "person_income", "loan_amnt")) {
      if (num_var %in% names(df)) {
        p <- ggplot(df, aes(x = .data[[cat_var]], y = .data[[num_var]])) +
          geom_boxplot(fill = "orange", alpha = 0.7) +
          labs(
            title = paste(num_var, "по", cat_var),
            subtitle = ifelse(num_var == "loan_int_rate",
              "После обработки пропусков", ""
            ),
            x = cat_var, y = num_var
          ) +
          theme_minimal() +
          theme(axis.text.x = element_text(angle = 45, hjust = 1))
        print(p)
        readline(prompt = "Нажмите Enter для продолжения")
      }
    }
  }
}

for (num_var in c(
  "loan_int_rate", "person_income",
  "loan_amnt", "person_age", "person_emp_length"
)) {
  p <- ggplot(df, aes(x = factor(loan_status), y = .data[[num_var]])) +
    geom_boxplot(fill = "lightblue", alpha = 0.7) +
    labs(
      title = paste("Распределение", num_var, "по статусу кредита"),
      x = "Статус кредита (0=хороший, 1=дефолт)",
      y = num_var
    ) +
    theme_minimal()
  print(p)
  readline(prompt = "Нажмите Enter для продолжения...")
}


# Выбираем только числовые колонки
num_cols <- df %>% select(where(is.numeric))

# Сводная таблица с описательной статистикой
desc_stats <- num_cols %>%
  pivot_longer(everything(), names_to = "variable") %>%
  group_by(variable) %>%
  summarise(
    mean = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    mode = {
      tbl <- table(value)
      as.numeric(names(tbl)[which.max(tbl)])
    },
    variance = var(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    min = min(value, na.rm = TRUE),
    max = max(value, na.rm = TRUE),
    q25 = quantile(value, 0.25, na.rm = TRUE),
    q75 = quantile(value, 0.75, na.rm = TRUE)
  )

print(desc_stats)


# Преобразование категориальных переменных
categorical_cols <- c(
  "person_home_ownership", "loan_intent",
  "loan_grade", "cb_person_default_on_file"
)

# Создаем dummy-переменные
df <- dummy_cols(df,
  select_columns = categorical_cols,
  remove_first_dummy = TRUE,
  remove_selected_columns = TRUE
)

# Разделение на признаки и целевую переменную
X <- df %>% select(-loan_status)
y <- df$loan_status

# Разделение на обучающую и тестовую выборки
set.seed(42)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- X[train_index, ]
x_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Подбор количества деревьев
estimators <- c(30, 50, 70, 90, 110)
for (i in estimators) {
  model <- randomForest(x = x_train, y = y_train, ntree = i)
  pred <- predict(model, x_test)
  acc <- mean(pred == y_test)
  cat(paste0("Деревьев: ", i, ", accuracy: ", round(acc, 4), "\n"))
}

# Обучение модели
best_ntree <- 70
final_model <- randomForest(x = x_train, y = y_train, ntree = best_ntree)
predictions <- predict(final_model, x_test)

# Оценка модели
conf_matrix <- table(Фактические = y_test, Предсказанные = predictions)
accuracy <- mean(predictions == y_test)

cat("\nИтоговая точность модели:", round(accuracy, 4), "\n")
cat("Матрица ошибок:\n")
print(conf_matrix)

# Визуализация матрицы ошибок
conf_matrix_melted <- melt(conf_matrix)

ggplot(
  conf_matrix_melted,
  aes(x = Предсказанные, y = Фактические, fill = value)
) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), color = "white", size = 6) +
  scale_fill_gradient(low = "#4393c3", high = "#2166ac") +
  labs(
    title = "Матрица ошибок",
    x = "Предсказанные значения",
    y = "Фактические значения"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

# Сравнение распределений
comparison_data <- data.frame(
  Тип = c(
    rep("Фактические", length(y_test)),
    rep("Предсказанные", length(predictions))
  ),
  Значение = c(as.character(y_test), as.character(predictions))
)

ggplot(comparison_data, aes(x = Значение, fill = Тип)) +
  geom_bar(position = "dodge", alpha = 0.8) +
  scale_x_discrete(labels = c("0" = "Без дефолта", "1" = "Дефолт")) +
  scale_fill_manual(values = c("#377eb8", "#e41a1c")) +
  labs(
    title = "Визуализация предсказаний модели",
    x = "Статус кредита",
    y = "Количество"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")



# Получение важности признаков
importance_df <- as.data.frame(final_model$importance)
importance_df$feature <- rownames(importance_df)
rownames(importance_df) <- NULL

# Сортировка по убыванию важности
importance_df <- importance_df[order(-importance_df$MeanDecreaseGini), ]

# Вывод таблицы важности
print(importance_df)

varImpPlot(final_model,
  main = "Важность признаков",
  type = 2
)
