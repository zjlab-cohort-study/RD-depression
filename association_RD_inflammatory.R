# ===============================
# LMM 分析：RD_label vs 13 种炎症因子（log+z-score）
# ===============================
# rm(list = ls())
# 加载必要包
library(lme4)        
library(lmerTest)     
library(dplyr)
library(effectsize)  
library(scales)       

# -------------------------------
# 1. 读取数据
# -------------------------------
data <- read.csv("input.csv", stringsAsFactors = TRUE)
out_file<-"output.csv"

# -------------------------------
# 2. 设置暴露、炎症因子、协变量
# -------------------------------
exposure <- "RD_early_label"

mediators <- c("Leukocyte", "Platelet_count", "Lymphocyte_count", "Monocyte_count",
               "Neutrophil_count", "Eosinophil_count", "Basophil_count",
               "Lymphocyte_percentage", "Monocyte_percentage", "Neutrophil_percentage",
               "Eosinophil_percentage", "Basophil_percentage", "CRP")

covariates <- c("age_group", "sex", "race_group", "TDI_group", "family_income",
                "college_group", "obesity_group", "sedentary_group",
                "smoking_status", "alcohol_intake")

random_effect <- "assessment_center"

# -------------------------------
# 3. 数据预处理：log(x+1) 和 z-score
# -------------------------------
for (mediator in mediators) {
  # log(x + 1)
  data[[paste0(mediator, "_log")]] <- log(data[[mediator]] + 1)
  # z-score 标准化
  data[[paste0(mediator, "_z")]] <- scale(data[[paste0(mediator, "_log")]])
}

# 使用处理后的变量
mediators_z <- paste0(mediators, "_z")

# -------------------------------
# 4. 循环每个炎症因子做 LMM
# -------------------------------
results <- data.frame(
  mediator = character(),
  beta_std = numeric(),
  cohen_d = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

for (mediator in mediators_z) {
  
  # 构建公式
  formula_str <- paste0(
    mediator, " ~ ", exposure, " + ", paste(covariates, collapse = " + "), 
    " + (1 | ", random_effect, ")"
  )
  
  # 拟合模型
  model <- lmer(as.formula(formula_str), data = data)
  
  # 提取标准化 beta 及 95% CI
  std_res <- standardize_parameters(model, ci = 0.95, exponentiate = FALSE)
  beta_std <- std_res %>% filter(Parameter == exposure) %>% pull(Std_Coefficient)
  ci_lower <- std_res %>% filter(Parameter == exposure) %>% pull(CI_low)
  ci_upper <- std_res %>% filter(Parameter == exposure) %>% pull(CI_high)
  
  # 转换为 Cohen's d
  cohen_d_val <- 2 * beta_std
  
  # 提取 p 值
  p_val <- summary(model)$coefficients[exposure, "Pr(>|t|)"]
  
  # 保存结果
  results <- rbind(results, data.frame(
    mediator = mediator,
    beta_std = beta_std,
    CI_lower = ci_lower,
    CI_upper = ci_upper,
    cohen_d = cohen_d_val,
    p_value = p_val,
    stringsAsFactors = FALSE
  ))
}

# -------------------------------
# 5. 多重比较校正（Bonferroni）
# -------------------------------
results$p_fdr <- p.adjust(results$p_value, method = "BH")
results$p_bonf <- p.adjust(results$p_value, method = "bonferroni")

# -------------------------------
# 6. 输出结果
# -------------------------------
print(results)
write.csv(results, out_file, row.names = FALSE)
