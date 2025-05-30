#!/bin/bash
#SBATCH --account=def-adurand #rrg-adurand
#SBATCH --output=/scratch/siham/slurm_%j.out  # Fichier de sortie dans le répertoire scratch
#SBATCH --error=/scratch/siham/mon_job_%j.err   # Fichier d'erreurs dans le répertoire scratch
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=5000M
#SBATCH --time=01:00:00
#SBATCH --mail-user=siham.si-hadj-mohand@sres.ulaval.ca
#SBATCH --mail-type=ALL
# ici on load les modules de compute canada dont on a besoin
module --force purge
module load StdEnv/2023 python/3.11.5
# Définir le répertoire temporaire personnalisé
# ici on vient activer l'environnement virtuel qui est stocke dans scratch
source ~/scratch/ENV/bin/activate
# pip install  -r requirements.txt
# ici on vient charger dans SLURM TMPDIR la base de donnees volumineuse et compressee, puis on la deco>
if [ "${DATA}" = "data" ]; then
    echo 'unzip data'
    mkdir -p $TMPDIR/mimic
    ls -l $TMPDIR/mimic
    mkdir -p $TMPDIR/data/output
    mkdir -p $TMPDIR/data2/output
    mkdir -p $TMPDIR/data3/output
    mkdir -p $TMPDIR/data4/output
    if tar xf ~/scratch/data.tar.gz -C $TMPDIR/mimic; then
         echo 'Data unzipped successfully'
    else
         echo 'Failed to unzip data'
         exit 1
    fi

    echo 'data unzipped'
fi
echo 'HZ: start python3 ./main.py ..at '; date
# ici on execute le code
python3 -u ./main.py &  # Mod>
python3 -u ./main2.py &  # Mo>
python3 -u ./main3.py &  # Mo>
python3 -u ./main4.py &  # Mo>
wait

# Copier les résultats dans /scratch sur le nœud de login
cp -r $TMPDIR/data/ ~/scratch/tmp/
cp -r $TMPDIR/data2/ ~/scratch/tmp2/
cp -r $TMPDIR/data3/ ~/scratch/tmp3/
cp -r $TMPDIR/data4/ ~/scratch/tmp4/

echo 'Les données ont été enregistrées avec succés'
