suppressMessages(library(pegas))
suppressMessages(library(vcfR))

args <- commandArgs(trailingOnly = TRUE)
# args[1]: vcf file
# args[2]: output file 

vcfR_hn_orivcf <- read.vcfR(args[1])

dna <- vcfR2DNAbin(vcfR_hn_orivcf, consensus = T, extract.haps = F)
hap <- haplotype(dna)
hap <- sort(hap, what = "labels")
hap.net <- haploNet(hap, getProb = FALSE)

sample_order <- attr(dna, "dimnames")[[1]]
hap_order <- attr(hap, "dimnames")[[1]]
detail_lis <- c()
for (i in seq_along(hap_order)){
    detail_lis <- c(detail_lis, paste(sample_order[attr(hap, "index")[[i]]], collapse = ","))
}
fra_hn_detail <- data.frame(hap_order, detail_lis)
write.table(fra_hn_detail, args[2], quote = F, row.names = F, col.names = F, sep = "\t")
