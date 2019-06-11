# One-hundred-species-classification
différentes méthodes de classification supervisée de 100 espèces de plantes utilisant les réseaux de neurones, les arbres de décision et une classification knn
Pre -traitements des donnees
REMARQUES
• Dans la famille Acer, la sous-famille « Campestre » manque une donne e de texture
• Toute autre sous famille d’especes contient 16 diffe rentes instances dans chaque
CONSEQUENCES
• La sous-famille « Campestre » est ignore e dans notre travail
SOLUTION PROPOSEE
. Nos donne es sont de ja e tiquete es donc notre classification sera supervise e
. Les donne es sont acceptables, le seul proble me qui se pose c’est le type des labels.
De notre repre sentation de chaque tuple de plantes, on a 192 donne es diffe rentes (merge, texture, Shape) d’ou on a choisi d’utiliser 192 neurones dans la couche initiale des inputs.
. Pour le nombre de couches, on commencera par utiliser seulement 1 seule couche cache e.
. On a aussi 99 classes de plantes diffe rentes, donc on aura besoin de 99 neurones comme sorties
Classification de donne es utilisant le classifieur Knn, et les arbres de de cision
MOTIVATION
Dans le but de comparaison entre l’utilisation des re seaux de neurones et d’autres types de classification vus en cours on a de cide de tester la classification de nous donne es utilisant le classifieur Knn et les arbres de de cision.
REMARQUE :
L’utilisation de re seau de neurones a donne e des re sultats meilleurs
