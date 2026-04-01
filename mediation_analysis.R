########################################################################
# - Exposure: binary 0/1
# - Mediators: list of continuous vars (will do log(x+1) + z)
# - Outcome: survival (time + event)
# - Bootstrap: fully re-fit a and b in each bootstrap sample
# - Multiple testing: both FDR (BH) 和 Bonferroni
#
########################################################################

library(dplyr)
library(survival)
library(broom)
setwd(dir = "workdir")

# -------------------------------
# 0. configs
# -------------------------------
data_path <- "input.csv"
exposure  <- "RD_early_label"        # <- 二分类 0/1 列名
time_var  <- "follow_time"     # <- 随访时间列名
event_var <- "depression_late_label"           # <- 结局事件列名（0/1）

# mediator：
mediators <- c(
  "Leukocyte","Platelet_count","Lymphocyte_count","Monocyte_count",
  "Neutrophil_count","Eosinophil_count","Basophil_count",
  "Lymphocyte_percentage","Monocyte_percentage","Neutrophil_percentage",
  "Eosinophil_percentage","Basophil_percentage","CRP"
)

# covariates:
covars <- c("age_group","sex","race_group","TDI_group","family_income",
            "college_group","obesity_group","sedentary_group",
            "smoking_status","alcohol_intake")

# Bootstrap 
B <- 2000   

# random seed
set.seed(12345)

# -------------------------------
# 1. 读入数据 & 基础检查
# -------------------------------
dat0 <- read.csv(data_path, stringsAsFactors = FALSE)

# 确保必须列存在
need_cols <- c(exposure, time_var, event_var, mediators, covars)
miss <- setdiff(need_cols, names(dat0))
if(length(miss) > 0) stop("缺少以下列：", paste(miss, collapse = ", "))

# 复制一份，后续都用 dat
dat <- dat0

# -------------------------------
# 2. 对所有 mediator 做 log(x+1) + z-score
# -------------------------------
mediators_logz <- paste0(mediators, "_logz")
for(i in seq_along(mediators)){
  m <- mediators[i]
  newn <- mediators_logz[i]
  
  if (is.factor(dat[[m]])) {
    dat[[m]] <- as.numeric(as.character(x))
  } else if (is.character(dat[[m]])) {
    dat[[m]] <- as.numeric(dat[[m]])
  }
  # 允许 NA 存在；对负数也做 log(x+1)（若负值存在请先处理）
  dat[[newn]] <- log(dat[[m]] + 1)
  dat[[newn]] <- as.numeric(scale(dat[[newn]], center = TRUE, scale = TRUE)[,1])
}

# -------------------------------
# 3. 逐个 mediator 执行完整中介分析并做 Bootstrap
#    返回：a, b, c_total(c), c_prime, ACME (obs), ACME CI (boot), ACME p (boot)
# -------------------------------
results <- list()

cat("开始逐个 mediator 的完全 bootstrap 中介分析：B =", B, "\n\n")

for(i in seq_along(mediators)){
  med_raw <- mediators[i]
  med <- mediators_logz[i]   
  cat("===== 处理 mediator:", med_raw, " (col:", med, ") =====\n")
  
  # 过滤必要变量的 NA（为原始完整样本做估计）
  df_full <- dat %>% 
    dplyr::select(all_of(c(exposure, time_var, event_var, covars, med))) %>%
    filter(!is.na(.data[[exposure]]), !is.na(.data[[time_var]]), !is.na(.data[[event_var]]))
  
  # 如果样本太少则跳过
  if(nrow(df_full) < 50){
    warning("样本量小于50，跳过 ", med_raw)
    next
  }
  
  # -------------------
  # a: lm(M ~ X + covars)
  # -------------------
  f_a <- as.formula(paste0(med, " ~ ", exposure, " + ", paste(covars, collapse = " + ")))
  fit_a <- tryCatch(lm(f_a, data = df_full), error=function(e) NULL)
  if(is.null(fit_a)){
    warning("a 路径 lm 拟合失败，跳过 ", med_raw); next
  }
  a_obs <- coef(fit_a)[exposure]
  if(is.na(a_obs)){ warning("a 估计为 NA，跳过 ", med_raw); next }
  
  # -------------------
  # c: 总效应 Cox (Y ~ X + covars)
  # -------------------
  f_c <- as.formula(paste0("Surv(", time_var, ",", event_var, ") ~ ", exposure, " + ", paste(covars, collapse = " + ")))
  fit_c <- tryCatch(coxph(f_c, data = df_full), error=function(e) NULL)
  if(is.null(fit_c)){
    warning("c (总效应) Cox 拟合失败，跳过 ", med_raw); next
  }
  coef_c <- coef(fit_c)[exposure]
  
  # -------------------
  # b & c': Cox (Y ~ X + M + covars)
  # -------------------
  f_b <- as.formula(paste0("Surv(", time_var, ",", event_var, ") ~ ", exposure, " + ", med, " + ", paste(covars, collapse = " + ")))
  fit_b <- tryCatch(coxph(f_b, data = df_full), error=function(e) NULL)
  if(is.null(fit_b)){
    warning("b/c' Cox 拟合失败，跳过 ", med_raw); next
  }
  b_obs <- coef(fit_b)[med]
  cprime_obs <- coef(fit_b)[exposure]
  
  # ACME 观察值
  ACME_obs <- as.numeric(a_obs * b_obs)
  
  # -------------------
  # Bootstrap：每次重抽样全部样本（with replacement），在 bootstrap 样本中重新拟合 a 和 b
  # -------------------
  cat("  Running bootstrap ...\n")
  ACME_boot <- numeric(0)
  valid_count <- 0
  # 预分配加速
  ACME_boot_vec <- rep(NA, B)
  
  for(bi in 1:B){
    # 抽样行号
    sidx <- sample.int(nrow(df_full), size = nrow(df_full), replace = TRUE)
    df_b <- df_full[sidx, , drop = FALSE]
    
    # 拟合 a (lm)
    fa_b <- tryCatch(lm(f_a, data = df_b), error=function(e) NULL)
    if(is.null(fa_b)) { ACME_boot_vec[bi] <- NA; next }
    a_b <- coef(fa_b)[exposure]
    if(is.na(a_b)) { ACME_boot_vec[bi] <- NA; next }
    
    # 拟合 b (cox)
    fb_b <- tryCatch(coxph(f_b, data = df_b), error=function(e) NULL)
    if(is.null(fb_b)) { ACME_boot_vec[bi] <- NA; next }
    # 若 b 的系数缺失
    b_b <- tryCatch(coef(fb_b)[med], error=function(e) NA)
    if(is.na(b_b)) { ACME_boot_vec[bi] <- NA; next }
    
    # 保存
    ACME_boot_vec[bi] <- as.numeric(a_b * b_b)
    valid_count <- valid_count + 1
  } # end bootstrap loop
  
  # 清理有效值
  ACME_boot_valid <- ACME_boot_vec[!is.na(ACME_boot_vec)]
  n_valid <- length(ACME_boot_valid)
  if(n_valid < max(50, floor(B*0.5))){
    warning(sprintf("Bootstrap 有效次数太低 (%d/%d) for %s — 结果可能不稳定", n_valid, B, med_raw))
  }
  
  # 计算 95% percentile CI 与 p-value (two-sided)
  if(n_valid > 0){
    CI_low  <- as.numeric(quantile(ACME_boot_valid, probs = 0.025, na.rm=TRUE))
    CI_high <- as.numeric(quantile(ACME_boot_valid, probs = 0.975, na.rm=TRUE))
    # 双侧 p-value：极端性基于绝对值
    p_boot <- (sum(abs(ACME_boot_valid) >= abs(ACME_obs)) + 1) / (n_valid + 1)
  } else {
    CI_low <- NA; CI_high <- NA; p_boot <- NA
  }
  
  # 中介比例（注意：c_total 可为 0）
  Prop_med <- ifelse(is.na(coef_c) | coef_c == 0, NA, ACME_obs / coef_c)
  
  # 存结果
  results[[med_raw]] <- list(
    mediator = med_raw,
    a = as.numeric(a_obs),
    b = as.numeric(b_obs),
    c_total = as.numeric(coef_c),
    c_prime = as.numeric(cprime_obs),
    ACME = as.numeric(ACME_obs),
    ACME_CI_low = CI_low,
    ACME_CI_high = CI_high,
    ACME_p = p_boot,
    ACME_boot_valid = n_valid,
    Prop_mediated = as.numeric(Prop_med)
  )
  
  cat(sprintf("  Done %s: a=%.4g, b=%.4g, ACME=%.4g, CI=[%.4g, %.4g], p=%.4g (valid %d/%d)\n",
              med_raw, a_obs, b_obs, ACME_obs, CI_low, CI_high, p_boot, n_valid, B))
  
} # end mediator loop

# -------------------------------
# 4. 聚合结果为 data.frame，并做多重检验校正（FDR & Bonferroni）
# -------------------------------
res_df <- do.call(rbind, lapply(results, function(x){
  data.frame(
    mediator = x$mediator,
    a = x$a, b = x$b,
    c_total = x$c_total,
    c_prime = x$c_prime,
    ACME = x$ACME,
    ACME_CI_low = x$ACME_CI_low,
    ACME_CI_high = x$ACME_CI_high,
    ACME_p = x$ACME_p,
    ACME_boot_valid = x$ACME_boot_valid,
    Prop_mediated = x$Prop_mediated,
    stringsAsFactors = FALSE
  )
}))

# 校正 p 值
res_df$ACME_p_FDR <- p.adjust(res_df$ACME_p, method = "BH")
res_df$ACME_p_Bonferroni <- p.adjust(res_df$ACME_p, method = "bonferroni")

# 保存 CSV
outname <- paste0("Mediation_bootstrap_results_B", B, ".csv")
write.csv(res_df, outname, row.names = FALSE)
cat("\n结果已保存到：", outname, "\n")



