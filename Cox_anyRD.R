# ============================================
#      Cox 回归脚本（含协变量）
# ============================================

library(survival)
library(dplyr)
library(readr)
library(broom)

# -----------------------------
# 输入/输出路径
# -----------------------------
input_file <- "input_file.csv"
output_file <- "output_file.csv"

# -----------------------------
# 定义变量
# -----------------------------
exposure <- "RD_label"
event <- "depression_label"
time <- "follow_time"

 categorical_vars <- c("age_group", "sex", "race_group") #model-1
# categorical_vars <- c("age_group", "sex", "race_group", "TDI_group", "obesity_group","sedentary_group", "college_group", "family_income","alcohol_intake", "smoking_status") #model-2
# categorical_vars <- c("age_group", "sex", "race_group", "TDI_group", "obesity_group","sedentary_group", "college_group", "family_income","alcohol_intake", "smoking_status",
#                       "diabetes_diagnosed_by_doctor","cancer_diagnosed_by_doctor","vascular_or_heart_problems") #model-3 add covar
# -----------------------------
# 读入数据并对follow_time进行筛选
# -----------------------------
data <- read_csv(input_file)
data_filt <- data %>% filter(.data[[time]] >= 10)
# -----------------------------
# 转为 factor（类别型变量）
# -----------------------------
data_filt[categorical_vars] <- lapply(data_filt[categorical_vars], factor)

# -----------------------------
# 构建公式
# -----------------------------
formula_str <- paste0(
  "Surv(", time, ", ", event, ") ~ ",
  exposure, " + ",
  paste(categorical_vars, collapse = " + ")
)
cox_formula <- as.formula(formula_str)

# -----------------------------
# 拟合 Cox 回归
# -----------------------------
cox_model <- coxph(cox_formula, data = data_filt)

# -----------------------------
# 提取 HR、95%CI、p 值
# -----------------------------
results <- tidy(cox_model, exponentiate = TRUE, conf.int = TRUE) %>% 
  select(term, estimate, conf.low, conf.high, p.value) %>%
  rename(
    HR = estimate,
    CI_lower = conf.low,
    CI_upper = conf.high,
    p = p.value
  )

# -----------------------------
# 输出结果到 CSV
# -----------------------------
print(results)
write_csv(results, output_file)

cat("Cox 回归结果已保存至：", output_file, "\n")
