
library(survival)
library(splines)
library(dplyr)
library(ggplot2)
library(broom)
library(car)
setwd(dir = "workdir")

data <- read.csv("iput.csv", stringsAsFactors = TRUE)

mediators <- c(
  "Leukocyte","Platelet_count","Lymphocyte_count","Monocyte_count",
  "Neutrophil_count","Eosinophil_count","Basophil_count",
  "Lymphocyte_percentage","Monocyte_percentage","Neutrophil_percentage",
  "Eosinophil_percentage","Basophil_percentage","CRP"
)

covars <- c("age_group","sex","race_group","TDI_group","family_income",
            "college_group","obesity_group","sedentary_group",
            "smoking_status","alcohol_intake")

time_col <- "follow_time"
event_col <- "depression_late_label"

run_continuous_spline <- function(df, time_var, event_var, mediator, covars, df_spline = 3) {
    df[[mediator]] <- log(df[[mediator]] + 1)
    df[[mediator]] <- scale(df[[mediator]], center=TRUE, scale=TRUE)[,1]
    df[[event_var]] <- as.numeric(df[[event_var]])

    # 构建公式
    spline_term <- paste0("ns(", mediator, ", df=", df_spline, ")")
    fml <- as.formula(
        paste0("Surv(", time_var, ", ", event_var, ") ~ ",
               spline_term, " + ", paste(covars, collapse = " + "))
    )

    # 拟合模型
    fit <- coxph(fml, data = df)

    # 抽取样条项的列名
    fit_coef_names <- names(coef(fit))
    spline_cols <- fit_coef_names[grepl(paste0("^ns\\(", mediator), fit_coef_names)]

    # ---- P-overall：整体非线性检验（所有样条项） ----
    L_overall <- matrix(0, nrow = length(spline_cols), ncol = length(fit_coef_names))
    rownames(L_overall) <- spline_cols
    colnames(L_overall) <- fit_coef_names
    for (i in seq_along(spline_cols)) {
        L_overall[i, spline_cols[i]] <- 1
    }
    p_overall <- linearHypothesis(fit, L_overall, test = "Chisq")$`Pr(>Chisq)`[2]

    # ---- P-nonlinear：非线性成分（除第一条外的样条项）----
    linear_term <- spline_cols[1]
    nonlinear_terms <- spline_cols[-1]

    L_nonlin <- matrix(0, nrow = length(nonlinear_terms), ncol = length(fit_coef_names))
    rownames(L_nonlin) <- nonlinear_terms
    colnames(L_nonlin) <- fit_coef_names
    for (i in seq_along(nonlinear_terms)) {
        L_nonlin[i, nonlinear_terms[i]] <- 1
    }
    p_nonlinear <- linearHypothesis(fit, L_nonlin, test = "Chisq")$`Pr(>Chisq)`[2]

    # ---- HR 曲线 ----
    x_seq <- seq(min(df[[mediator]], na.rm=TRUE),
                 max(df[[mediator]], na.rm=TRUE),
                 length=100)

    # prediction data frame MUST contain mediator and ALL covariates
    df_pred <- data.frame(mediator_value = x_seq)
    names(df_pred)[1] <- mediator

    # covariates 固定为 reference，例如最常见的类别
    for (v in covars) {
        if (is.numeric(df[[v]])) {
            df_pred[[v]] <- median(df[[v]], na.rm=TRUE)
        } else {
            df_pred[[v]] <- names(sort(table(df[[v]]), decreasing = TRUE))[1]
        }
    }

    pred0 <- predict(fit, newdata = df_pred, type = "lp", se.fit = TRUE)
    df_hr <- data.frame(
        x = x_seq,
        HR = exp(pred0$fit),
        CI_lower = exp(pred0$fit - 1.96 * pred0$se.fit),
        CI_upper = exp(pred0$fit + 1.96 * pred0$se.fit)
    )

    return(list(
        df_hr = df_hr,
        p_overall = p_overall,
        p_nonlinear = p_nonlinear,
        model = fit
    ))
}

###############################################
## 2. 批处理分析：continuous + tertile
###############################################
all_cont <- list()
results_tertile_all <- data.frame()

for (med in mediators) {
  message("=== Running: ", med, " ===")
  
  ##############################
  ## 连续样条
  ##############################
  res <- tryCatch({
    run_continuous_spline(
      df = data,
      time_var = time_col,
      event_var = event_col,
      mediator = med,
      covars = covars
    )
  }, error=function(e){
    message("Spline failed: ", med, " :: ", e$message)
    return(NULL)
  })
  
  if (!is.null(res)) {
    df_hr <- res$df_hr
    df_hr$mediator <- med
    
    # write.csv(df_hr, paste0(med, "_Continuous_HRcurve.csv"), row.names = FALSE)
    
    pdf(paste0(med, "_Continuous_HRcurve.pdf"), width=6, height=4)
    print(
      ggplot(df_hr, aes(x = x, y = HR)) +
        geom_line(color="steelblue") +
        geom_ribbon(aes(ymin = CI_lower, ymax = CI_upper), alpha = 0.3) +
        labs(title = paste0(med, " Continuous HR Curve"),
             subtitle = paste0("P-overall=", signif(res$p_overall,3),
                               " | P-nonlinear=", signif(res$p_nonlinear,3)),
             x = med, y = "HR") +
        coord_cartesian(ylim = c(0.8, max(df_hr$CI_upper, na.rm = TRUE))) +
        theme_bw(base_size = 14)
    )
    dev.off()
    
    all_cont[[med]] <- df_hr
  }
  
  ##############################
  ## Tertile
  ##############################
  dat <- data %>% filter(!is.na(.data[[med]]))
  
  q_ter <- unique(quantile(dat[[med]], probs=c(0,1/3,2/3,1)))
  if (length(q_ter) == 4) {
    dat[[paste0(med,"_tertile")]] <- cut(dat[[med]], breaks=q_ter,
                                         include.lowest=TRUE, labels=c("T1","T2","T3"))
    
    formula_ter <- as.formula(
      paste0("Surv(", time_col,",", event_col,") ~ ",
             med,"_tertile + ", paste(covars, collapse="+"))
    )
    
    cox_ter <- suppressWarnings(coxph(formula_ter, data=dat))
    tidy_ter <- tidy(cox_ter, exponentiate=TRUE, conf.int=TRUE)
    
    tidy_ter <- tidy_ter %>%
      filter(grepl(paste0(med,"_tertile"), term)) %>%
      mutate(
        level = gsub(paste0(med,"_tertile"), "", term),
        mediator = med,
        HR = estimate,
        CI_lower = conf.low,
        CI_upper = conf.high,
        p_value = p.value
      ) %>%
      select(mediator, level, HR, CI_lower, CI_upper, p_value)
    
    ref_row <- data.frame(mediator=med, level="T1", HR=1, CI_lower=1, CI_upper=1, p_value=NA)
    tidy_ter2 <- rbind(ref_row, tidy_ter)
    tidy_ter2$level <- factor(tidy_ter2$level, levels=c("T1","T2","T3"))
    
    results_tertile_all <- rbind(results_tertile_all, tidy_ter2)
    
    pdf(paste0(med, "_Tertile_Bar.pdf"), width=6, height=4)
    print(
      ggplot(tidy_ter2, aes(x=level, y=HR)) +
        geom_col(fill="steelblue", width = 0.5) +
        geom_errorbar(aes(ymin=CI_lower, ymax=CI_upper), width=0.1) +
        geom_hline(yintercept=1, linetype="dashed", color="red") +
        labs(title=paste0(med," Tertile Cox"), x="Tertile", y="HR (ref=T1)") +
        coord_cartesian(ylim = c(0.75, 1.25)) +   # 固定 y 轴范围
        theme_bw(base_size = 14)
    )
    dev.off()
  }
}


###############################################
## 3. 汇总输出
###############################################
# if(length(all_cont) > 0){
#   write.csv(do.call(rbind, all_cont), "Cox_Continuous_All.csv", row.names = FALSE)
# }

if(nrow(results_tertile_all) > 0){
  write.csv(results_tertile_all, "Cox_Tertile_All.csv", row.names = FALSE)
}

message("=== All Processing Finished ===")
