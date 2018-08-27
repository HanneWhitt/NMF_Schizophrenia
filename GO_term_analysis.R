library('org.Hs.eg.db')
library('clusterProfiler')
library('tools')

GO_term_analysis_all_csvs <- function(folder_of_csvs){
  setwd(folder_of_csvs)
  for (file in list.files()){
    if (file_ext(file) == "csv" && !grepl('GO_results', file)){
      cat(paste('\n\nPerforming GO term analysis:', file))
      genes = read.csv(file, header = FALSE)$V1
      for (ont in c('MF', 'BP', 'CC')){
        go_results <- enrichGO(gene = genes, OrgDb = org.Hs.eg.db, 
                              keyType = 'ENSEMBL', ont = ont, pAdjustMethod = 'bonferroni', 
                              pvalueCutoff = 0.01, qvalueCutoff = 0.05)
        output_filename <- paste(file_path_sans_ext(file), '_GO_results_', ont, '.csv', sep = '')
        write.csv(as.data.frame(go_results), output_filename)
        cat(paste('\n', ont, 'results saved to', output_filename))
      }
    }
  }
}
  