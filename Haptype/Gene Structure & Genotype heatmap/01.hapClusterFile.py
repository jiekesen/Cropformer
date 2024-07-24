import ujson
import os
import random
import traceback

Genelist = ["Zm00001d008941", "Zm00001d011956"]

def random_str(num):
    H = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    salt = ''
    for i in range(num):
        salt += random.choice(H)
    return salt

def GenerateTree(GeneID):
    token = random_str(5)

    shell = "Rscript hap.R " + GeneID + ".vcf tmp/" + token
    os.system(shell)

    with open("tmp/" + token) as f:
        lines = f.readlines()
    HapInfo = []
    for line in lines:
        ele = line.replace("\n", "").split("\t")
        SMlst = ele[1].split(",")
        if len(SMlst) <= 30:
            continue
        HapInfo.append([ele[0], SMlst])
    with open("HapInfo/" + GeneID + ".txt", "w") as f:
        for Hap in HapInfo:
            f.write(Hap[0] + "\t" + ",".join(Hap[1]) + "\n")

    tree = {"1":{}}
    vcfidlst = []
    for i in range(len(HapInfo)):
        tree["1"][str(i+1)] = HapInfo[i][1]
        vcfidlst += HapInfo[i][1]
    # print(tree)

    Data = {
        'status': 'succeed',
        'treeData': {
            'name': 'Older Ancestor',
            'parent': 'null',
            'color': 'white',
            'hapcount': len(HapInfo),
            'type': 'OA',
            'children': []
        },
        'chromData': [673, 317, [[30, '#f00']]],
        'geneData': {
            'A': [],
            'B': {},
            'C': {},
            'D': {}
        }
    }
    for aHap in tree:
        aHapinstance = {
            'name': 'H.' + aHap,
            'type': 'aHap',
            'color': 'red',
            'children': []
        }
        for Hap in tree[aHap]:
            HapInstance = {
                'name': 'H.' + aHap + '.' + Hap,
                'type': 'Hap',
                'color': 'blue',
                'SMCount': str(len(tree[aHap][Hap])),
                'SMlst': "."
            }
            aHapinstance["children"].append(HapInstance)
        Data["treeData"]["children"].append(aHapinstance)

    with open("plotdata/" + GeneID + ".Tree.o", "w") as f:
        f.write("Phase1Callback(" + str(Data).replace('"', "'") + ")")

    with open("tmp/" + token + ".SMlst.txt", "w") as f:
        f.write("\n".join(vcfidlst) + "\n")
    shell = "bcftools view " + GeneID + ".vcf -S tmp/" + token + ".SMlst.txt  | bcftools query -f '%CHROM\\t%POS\\t%ALT[\\t%GT]\\n'"
    shellstdout = os.popen(shell).readlines()
    validposcount = 0
    SNPAnnDic = []
    SNPGTlst = []
    for posline in shellstdout:
        ele = posline.strip().split("\t")
        try:
            validposcount += 1
            # SNPAnnDic[ele[0] + ":" + ele[1]] = ele[0] + ":" + ele[1] + "|" + ("HIGH" if keepflag else ("MODERATE" if keepflag2 else "."))
            SNPAnnDic.append(ele[0] + ":" + ele[1] + "|.|" + ele[0] + ":" + ele[1])
            # print(SNPAnnDic)
            GTDic = {}
            for j in range(3, len(ele)):
                GTDic[vcfidlst[j-3]] = ele[j]
            SNPGTlst.append(GTDic)
        except Exception as e:
            traceback.print_exc()
            print(shell)
            exit(1)
    if validposcount == 0:
        print("No vaild SNPs in current gene")
        exit(1)


    plotdata = []
    Hapinfo = []
    for aHap in tree:
        for Hap in tree[aHap]:
            Hapinfo.append(Hap)
            dataobj = []
            for j in range(len(SNPGTlst)):
                count = [0, 0, 0, 0] # for -1, 0, 1, 2
                for SM in tree[aHap][Hap]:
                    if SNPGTlst[j][SM] == "./.":
                        count[0] += 1
                    elif SNPGTlst[j][SM] == "0/0":
                        count[1] += 1
                    elif SNPGTlst[j][SM] == "0/1" or SNPGTlst[j][SM] == "1/0" :
                        count[2] += 1
                    elif SNPGTlst[j][SM] == "1/1":
                        count[3] += 1
                if count[0] == max(count):
                    dataobj.append("-1")
                elif count[1] == max(count):
                    dataobj.append("0")
                elif count[2] == max(count):
                    dataobj.append("1")
                elif count[3] == max(count):
                    dataobj.append("2")
            plotdata.append(dataobj)



    shell = "zgrep " + GeneID + " Zea_mays.B73_RefGen_v4.48.gff3.gz"
    shellstdout = os.popen(shell).read().replace("\t", "\\t").replace("\n", "\\n")


    s2 = "Phase3Callback({"
    s2 += "'status': 'succeed',"
    s2 += "'title': '" + GeneID + "',"
    s2 += "'plotdata': " + str(plotdata).replace('"', "'") + ","
    s2 += "'Hapinfo': " + str(Hapinfo).replace('"', "'") + ","
    s2 += "'SNPinfo': " + str(SNPAnnDic).replace('"', "'") + ","
    s2 += "'gffinfo': '" + shellstdout + "'})"

    with open("plotdata/" + GeneID + ".Gene.o", "w") as f:
        f.write(s2)

for GeneID in Genelist:
    GenerateTree(GeneID)