# ==========================
# 倾向评分匹配 + 平衡性 Love Plot
# ==========================

# --- load packages ---
library(MatchIt)
library(dplyr)
library(readr)
library(cobalt)  
library(ggplot2)

# --- 输入文件 ---
input_file <- "input.csv"
df <- read_csv(input_file)

output_file <- "output.csv"

# --- 暴露变量 ---
exposure <- "RD_early_label"
df[[exposure]] <- factor(df[[exposure]], levels = c(0,1))

# --- 协变量 ---
covariates <- c("age_group", "sex", "race_group", "TDI_group", 
                "family_income", "college_group", "obesity_group", 
                "sedentary_group", "smoking_status", "alcohol_intake")

for(cv in covariates) df[[cv]] <- as.factor(df[[cv]])

# --- 匹配前 RD=1 样本数 ---
n_RD1_before <- sum(df[[exposure]] == 1)
cat("匹配前 RD=1 样本数:", n_RD1_before, "\n")

# --- 倾向评分匹配（nearest neighbor） ---
formula_str <- as.formula(paste(exposure, "~", paste(covariates, collapse = " + ")))

set.seed(42)
m.out <- matchit(
  formula_str,
  data = df,
  method = "nearest",
  distance = "logit",
  replace = FALSE,
  caliper = 0.02
)

# --- 匹配后的数据 ---
matched_df <- match.data(m.out)
n_matched <- nrow(matched_df)
cat("匹配后总样本数:", n_matched, "\n")
table_RD <- table(matched_df[[exposure]])
cat("匹配后 RD=1 / RD=0 样本数:\n")
print(table_RD)

# --- 输出匹配数据 ---
write.csv(matched_df, output_file, row.names = FALSE)

# --- 平衡性 summary ---
sum_out <- summary(m.out)
# --- 提取匹配前 SMD ---
smd_before <- as.data.frame(sum_out$sum.all[, c("Std. Mean Diff.")])
colnames(smd_before) <- "SMD_before"

# --- 提取匹配后 SMD ---
smd_after <- as.data.frame(sum_out$sum.matched[, c("Std. Mean Diff.")])
colnames(smd_after) <- "SMD_after"

# --- 获取协变量名称 ---
covariate_names <- rownames(smd_before)

# --- 合并 ---
smd_combined <- cbind(Variable = covariate_names, smd_before, smd_after)
write.csv(smd_combined, "matched_RD_balance_summary.csv", row.names = FALSE)

# --- Love plot（匹配前后 SMD 可视化） ---
lp<-love.plot(
  m.out, 
  stats = "mean.diffs",       # 用标准化均差（SMD）
  threshold = 0.02,            # 红色虚线 0.1
  var.order = "unadjusted",   # 按匹配前 SMD 排序
  abs = TRUE,                 # 取绝对值
  line = TRUE,
  stars = "raw",
  shapes = c("circle", "triangle"),
  colors = c("red", "blue")
)
print(lp)


