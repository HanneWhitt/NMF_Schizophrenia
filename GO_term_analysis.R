library('org.Hs.eg.db')
library('clusterProfiler')
library('tools')
library('tidyr')

GO_term_analysis_all_csvs <- function(folder_of_csvs){
  setwd(folder_of_csvs)
  for (file in list.files()){
    if (file_ext(file) == "csv" && !grepl('GO_results', file)){
      cat(paste('\n\nPerforming GO term analysis:', file))
      genes <- read.csv(file, header = FALSE)
      rank = extract_numeric(regmatches(file, gregexpr("r[[:digit:]]+.csv", file)))
      total_genes = dim(genes)[1]
      number_top_genes = ceiling((total_genes/rank)*2)
      cat(paste('Using number of top genes: ', number_top_genes))
      genes <- genes[order(-genes$V2),]
      genes_for_GO <- genes$V1[1:number_top_genes]
      print(dim(genes_for_GO))
      for (ont in c('MF', 'BP', 'CC')){
        go_results <- enrichGO(gene = genes_for_GO, OrgDb = org.Hs.eg.db, 
                              keyType = 'ENSEMBL', ont = ont, pAdjustMethod = 'bonferroni', 
                              pvalueCutoff = 0.01, qvalueCutoff = 0.05)
        output_filename <- paste(file_path_sans_ext(file), '_GO_results_', ont, '.csv', sep = '')
        write.csv(as.data.frame(go_results), output_filename)
        cat(paste('\n', ont, 'results saved to', output_filename))
      }
    }
  }
}


final_results_folder <- 'C:/Users/hanne/Documents/PROJECT/Project Data/final_results/'
for (subfolder in list.files(final_results_folder)){
  print(paste('GO TERM ANALYSIS - FOLDER', subfolder))
  top_metagenes_folder <- paste(final_results_folder, subfolder, 
                                '/top_genes_significant_metagenes', sep = "")
  GO_term_analysis_all_csvs(top_metagenes_folder)
}
     
     