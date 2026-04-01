
library(tidyverse)
library(survival)
library(broom)
library(ggplot2)

# -------------------------------
# 1. 读取数据
# -------------------------------
df <- read_csv("input.csv", show_col_types = FALSE)
df <- df %>%
  filter(
    follow_time > 5,
    multi_RD_label %in% c(0, 1, 2, 3)
  )

covars <- c("age_group", "sex", "race_group", "TDI_group",
            "obesity_group", "sedentary_group", "college_group",
            "family_income", "alcohol_intake", "smoking_status")
covars_formula <- paste(covars, collapse = " + ")

# -------------------------------
# 2. 创建变量
# -------------------------------
df$exp_num <- df$multi_RD_label           # 0-3 numeric
df$exp_fac <- factor(df$multi_RD_label, levels = c(0, 1, 2, 3))   # 0-3 factor

# -------------------------------
# 3. 线性趋势 Cox 模型
# -------------------------------
form_lin <- as.formula(
  paste0("Surv(follow_time, depression_late_label) ~ exp_num + ", covars_formula)
)
fit_lin <- coxph(form_lin, data = df)
res_lin <- tidy(fit_lin, exponentiate = TRUE, conf.int = TRUE) %>%
  filter(term == "exp_num")

# -------------------------------
# 4. 分类 Cox 模型
# -------------------------------
form_cat <- as.formula(
  paste0("Surv(follow_time, depression_late_label) ~ exp_fac + ", covars_formula)
)
fit_cat <- coxph(form_cat, data = df)

# 分类模型结果，包括0作为 reference
res_cat <- tidy(fit_cat, exponentiate = TRUE, conf.int = TRUE) %>%
  mutate(category = as.numeric(str_extract(term, "\\d+"))) %>%
  arrange(category)

# -------------------------------
# 5. 分类模型整体显著性 (P-overall)
# -------------------------------
p_overall_val <- summary(fit_cat)$sctest["pvalue"]
p_overall <- ifelse(p_overall_val < 0.001, "<0.001", signif(p_overall_val,3))

# -------------------------------
# 6. 线性 vs 分类模型 LRT (P-nonlinear)
# -------------------------------
p_nonlinear_val <- as.numeric(anova(fit_lin, fit_cat, test="LRT")$`Pr(>|Chi|)`[2])
p_nonlinear <- ifelse(p_nonlinear_val < 0.001, "<0.001", signif(p_nonlinear_val,3))

# -------------------------------
# 7. 保存结果 CSV
# -------------------------------
write_csv(res_lin, "Cox_linear_trend.csv")
write_csv(res_cat, "Cox_categorical_results.csv")

lrt_df <- tibble(
  Comparison = "Linear vs Categorical",
  P_overall = p_overall,
  P_nonlinear = p_nonlinear
)
write_csv(lrt_df, "Cox_LRT_results.csv")

# -------------------------------
# 8. 绘制 HR 曲线（0-3 类别，0 作为 reference=1）
# -------------------------------
# 手动加入 reference = 0
ref_row <- tibble(
  term = "exp_fac0",
  estimate = 1,
  conf.low = 1,
  conf.high = 1,
  category = 0
)
plot_df <- bind_rows(ref_row, res_cat) %>% arrange(category)

# 绘图
p <- ggplot(plot_df, aes(x = category, y = estimate)) +
  #  (95% CI)
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high), fill = "lightblue", alpha = 0.3) +
  # 绘制深蓝色折线
  geom_line(color = "darkblue", size = 1) +
  # 绘制HR点
  geom_point(color = "darkblue", size = 3) +
  # X轴从0开始
  scale_x_continuous(breaks = 0:3) +
  ylab("Hazard Ratio (ref=0)") +
  xlab("Number of diseases") +
  theme_bw(base_size = 14) +
  # 标注 P-overall 和 P-nonlinear
  annotate("text", x = 2, y = max(plot_df$conf.high)*1.05,
           label = paste0("P-overall=", p_overall,
                          ", P-nonlinear=", p_nonlinear),
           size = 5)

# 保存矢量图
ggsave("HR_curve_multi_RD.pdf", p, width = 7, height = 5)
