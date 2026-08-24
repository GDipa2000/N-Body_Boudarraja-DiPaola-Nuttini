#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "force.hpp"
#include "particle.hpp"
#include "quadtreeNode.hpp"

#include "fileUtilities.hpp"
#include "simulationUtilities.hpp"
#include "treeUtilities.hpp"

/**
 * @brief Parallel merge a vector of QuadtreeNode roots into a single QuadtreeNode.
 * Based on the number of subtrees, the function merges four subtrees at a time in a single one;
 * recursively, it merges the obtained subtrees until it reaches the root of the tree, which is then returned.
 *
 * @tparam Dimension Number of dimensions of the simulation
 * @param roots Reference to the vector of subtrees that need to be merged into a single tree
 * @param subRegionsDimension Dimension of the subregions in which the simulation area is divided
 * @param num_threads Number of threads that are used for the parallelization
 * @return Unique pointer to the root of the merged tree
 **/
template <size_t Dimension>
std::unique_ptr<QuadtreeNode<Dimension>> merging(std::vector<std::unique_ptr<QuadtreeNode<Dimension>>>* roots,
                                                 double subRegionsDimension, int num_threads) {
    int size = (*roots).size();  // always 4^n elements
    int k = 4; // multiplier to get the right index of the vector of roots
    int y = 1; 
    double x = 0.0; // x-coordinate of the center of the merged node
    double z = 0.0; // z-coordinate of the center of the merged node
    int l = 0;// level of the tree, used to calculate the center of the merged node

    std::array<std::unique_ptr<QuadtreeNode<Dimension>>, 4> temp_roots;

    while (size >= 4) {
        if (size < num_threads) {
            num_threads = size;
            omp_set_num_threads(num_threads);
        }

        #pragma omp parallel for private(temp_roots) shared(k, y) schedule(static, size / num_threads)
        for (int i = 0; i < size / 4; i++) {
            for (int j = 0; j < 4; j++) {
                int index = k * i + j * y;
                swap(temp_roots[j], (*roots)[index]);
            }

            (*roots).at(k * i) = std::move(mergeRoots(&temp_roots, subRegionsDimension * (l + 1)));
        }

        k = k * 4;
        y = y * 4;
        l++;
        size /= 4;  // Reduce the size of the vector by a factor of 4
    }


    return std::move((*roots)[0]);
}

/**
 * @brief Function that creates the root of the main quadtree and then starts working on merging the subtrees: 
 * if the root is a leaf, it checks if it contains a particle: if so, it means that that subtree contains only a particle, 
 * so it uses the normal insert of Quadtree class to inserts it into the tree; otherwise, the subtree has more than one particle,
 * so the root must be inserted as a child node, so it calls the function insertNode to take care of it.
 * 
 * @tparam Dimension Number of dimension of the simulation
 * @param particles Reference to a vector of unique pointers to the roots of the subtrees
 * @param subRegionsDimension Dimension of the subregions in which the simulation area is divided
 * @return Unique pointer to the root of the merged tree
 * **/
template <size_t Dimension>
std::unique_ptr<QuadtreeNode<Dimension>> mergeRoots(std::array<std::unique_ptr<QuadtreeNode<Dimension>>, 4>* roots,
                                               double subRegionsDimension) {
    double x, y; //dichiaro le coordinate del centro del nodo che sarà creato come radice del quadtree

    std::unique_ptr<QuadtreeNode<Dimension>> intNode;//Nodo intermedio che sarà creato come radice del quadtree
    double totx = 0.0; //variabile che conterrà la somma delle coordinate x dei centri figli
    double toty = 0.0; //variabile che conterrà la somma delle coordinate y dei centri figli
    //calcolo le coordinate del centro del nodo che sarà creato come radice del quadtree
    for (int i = 0; i < 4; i++) {
        totx += (*roots)[i]->getCenter()[0];//c
        toty += (*roots)[i]->getCenter()[1];
    }
    //il centro dell'albero risultante è dato dalla media dei centri dei quattro sottoalberi
    x = totx / 4.0;
    y = toty / 4.0;
    //creo il nodo intermedio che sarà la radice del quadtree che avrà ovviamnete 
    //larghezza doppia rispetto agli alberi che saranno inseriti come figli
    intNode = std::make_unique<QuadtreeNode<Dimension>>(x, y, 2 * subRegionsDimension, 20);
    //per l'aggregazione dei sottoalberi, se il nodo intermedio è una foglia, significa che 
    //il sottoalbero contiene un solo corpo, quindi lo inserisco nel quadtree come un normale corpo,
    //altrimenti il sottoalbero contiene più corpi, quindi il nodo intermedio deve essere inserito come nodo interno
    for (int j = 0; j < 4; j++) {
        if ((*roots)[j]->isLeaf()) {
            if ((*roots)[j]->getParticle() != nullptr) {
                intNode->insert((*roots)[j]->getParticle());
            }
        } else {
            //da aggiungere il controllo caso nullPtr------------------------------------->CONTROLLA!!
            intNode->insertNode(std::move((*roots)[j]));
        }
    //La variabile regions serve a memorizzare le coordinate dei sottoalberi che saranno inseriti
    //come figli del nodo intermedio,in modo da poter calcolare le coordinate del centro del nodo 
    //intermedio in base alle coordinate dei sottoalberi che saranno inseriti come figli
    std::vector<std::array<double, 2>> regions = getSubRegionsCoordinates<Dimension>(4, 2 * subRegionsDimension);
    }

    // if(intNode != nullptr)
    //     intNode->printTree();
    //ritorno il nodo intermedio che sarà la radice del quadtree
    return intNode;
}

/**
 * @brief Parallel function that divides the simulation area into subregions to be assigned to each thread, in order for the 
 * function CreateQuadTreeParallel to start creating the subtrees; then it merges them in a signle tree calling function 
 * merging and returns the root of it.
 * 
 * @tparam Dimension Number of dimensions of the simulation
 * @param particles Reference to the vector of particles in the simulation 
 * @param dimSimulationArea Dimension of the simulation area
 * 
*/
template <size_t Dimension>
std::unique_ptr<QuadtreeNode<Dimension>> generateTreeParallel(std::vector<Particle<Dimension>>* particles,
                                                              double dimSimulationArea, int num_threads) {
                                                        
    int num_quad = pow(4, ceil(log(num_threads) / log(4)));
    std::unique_ptr<QuadtreeNode<Dimension>> treeRoot;

    std::vector<std::array<double, 2>> regions = getSubRegionsCoordinates<Dimension>(num_quad, dimSimulationArea);
    std::array<double, 2> subRegionsDimension = getSubRegionsDimension<Dimension>(num_quad, dimSimulationArea);

    std::vector<std::unique_ptr<QuadtreeNode<Dimension>>> roots;
    roots.reserve(num_quad);

    for (int i = 0; i < num_quad; i++) {
        roots.push_back(nullptr);
    }

    #pragma omp parallel for shared(particles, regions)
    for (int i = 0; i < num_quad; i++) {
        (roots).at(i) =
            std::move(createQuadTreeParallel(*particles, regions[i], subRegionsDimension[0], dimSimulationArea));
    }

    treeRoot = merging(&roots, subRegionsDimension[0], num_threads);

    omp_set_num_threads(num_threads);

    return treeRoot;
}

/**
 * @brief Parallel function that determines the simulation boundaries, creates the root of the tree and then inserts each particle into the tree.
 * It checks if the particle is inside the boundaries of the simulation area (standard case) and, if so, inserts it into the tree. 
 * It also handles the case in which the particle is on the boundary of the simulation area.
 * 
 * @tparam Dimension Number of dimensions of the simulation
 * @param particles Reference to the vector of particles in the simulation
 * @param center Center of the simulation area
 * @param width Width of the simulation area
 * @param dimSimulationArea Dimension of the simulation area 
 * @return Unique pointer to the root of the tree
*/
template <size_t Dimension>
std::unique_ptr<QuadtreeNode<Dimension>> createQuadTreeParallel(std::vector<Particle<Dimension>>& particles,
                                                                std::array<double, Dimension> center, double width,
                                                                double dimSimulationArea) {
    //verifica se il vettore di particelle è vuoto
     if (particles.empty()) {
        return nullptr;
    }
    //calcolo delle coordinate dei bordi dell'area di simulazione (NEL CASO PARALLELO QUESTI PARAMETRI VARIANO)
    double minX = center[0] - width * 0.5;
    double maxX = center[0] + width * 0.5;
    double minY = center[1] - width * 0.5;
    double maxY = center[1] + width * 0.5;
    //creo quindi la radice del sottoalbero harcodando la profondità massima del sottoalbero a 20
    std::unique_ptr<QuadtreeNode<Dimension>> root =
        std::make_unique<QuadtreeNode<Dimension>>(center[0], center[1], width, 20);
    //Per evitare che una particella finisca in più quadranti o nessun quadrante viene fatto in  modo
    //tale che una particella presente sul bordo superiore o destro del quadrante non venga inserita in quel quadrante, 
    //ma in quello adiacente
    //CONDIZIONE BORDO IN BASSO A DESTRA
    if (maxX == dimSimulationArea / 2 && minY == -dimSimulationArea / 2) {
        //itero sulle particelle e verifico se sono all'interno dei bordi dell'area di simulazione,
        //in tal caso le inserisco nel sottoalbero
        for (auto& particle : particles) {
            if (particle.getPos()[0] >= minX && particle.getPos()[0] <= maxX && particle.getPos()[1] <= maxY &&
                particle.getPos()[1] >= minY) {
                std::shared_ptr<Particle<Dimension>> p = std::make_shared<Particle<Dimension>>(particle);
                root->insert(p);
            }
        }
    //BORDO INFERIORE
    } else if (minY == -dimSimulationArea / 2) {
        for (auto& particle : particles) {
            if (particle.getPos()[0] >= minX && particle.getPos()[0] < maxX && particle.getPos()[1] <= maxY &&
                particle.getPos()[1] >= minY) {
                std::shared_ptr<Particle<Dimension>> p = std::make_shared<Particle<Dimension>>(particle);
                root->insert(p);
            }
        }
    //BORDO DESTRO
    } else if (maxX == dimSimulationArea / 2) {
        for (auto& particle : particles) {
            if (particle.getPos()[0] >= minX && particle.getPos()[0] <= maxX && particle.getPos()[1] <= maxY &&
                particle.getPos()[1] > minY) {
                std::shared_ptr<Particle<Dimension>> p = std::make_shared<Particle<Dimension>>(particle);
                root->insert(p);
            }
        }
    }
    //caso standard in cui la particella si trova dentro ai bordi dell'area di simulazione e quindi viene inserita nel quadrante 
    else {  // standard case
        for (auto& particle : particles) {
            if (particle.getPos()[0] >= minX && particle.getPos()[0] < maxX && particle.getPos()[1] <= maxY &&
                particle.getPos()[1] > minY) {
                std::shared_ptr<Particle<Dimension>> p = std::make_shared<Particle<Dimension>>(particle);
                root->insert(p);
            }
        }
    }

    return root;
}

/**
 * @brief Function that calculates the net force acting on a particle; starting from the root of the quadtree, the function follows this procedure:
 * if the current node is an external node (and it is not body p), calculate theforce exerted by the current node on p, and add this amount to b’s 
 * net force. Otherwise, calculate the ratio s/d. If s/d < θ, treat this internal node as a single body, and calculate the force it exerts on body 
 * p, and add this amount to p’s net force. Finally, it run the procedure recursively on each of the current node’s children.
 * 
 * @tparam Dimension Number of dimensions of the simulation
 * @param node Reference to the root of the quadtree
 * @param p Reference to the particle on which the net force is calculated
 * @param theta Parameter that determines the accuracy of the approximation
 * @param f Reference to the Force object responsible for calculating particle interactions.
*/
template <size_t Dimension>
void calculateNetForceQuadtree(const std::unique_ptr<QuadtreeNode<Dimension>>& node,
                               std::shared_ptr<Particle<Dimension>> p, double theta, Force<Dimension>& f,
                               double dimSimulationArea, double softening) {
    /*
        QUESTA FUNZIONE VA UTILIZZATA SOLO AD ALBERO CALCOLATO
    */
    double s, d;

    std::array<double, Dimension> force_qk;
    std::unique_ptr<Particle<Dimension>> approxParticle;
    //Funzione fondamentale perché tutti i quadranti vuoti  sono rappresentati da nodi nulli
    if (node == nullptr) {
        return;
    }
    //gestione della condizione dei casi di collisione tra particella e area di bordo della simulazione
    if (p->hitsBoundary(dimSimulationArea)) {
        p->manageCollision(*p, dimSimulationArea);
    }
    //questi due if annidati servono a gestire il caso in cui il nodo corrente sia una foglia e contenga solo un corpo
    if (node->isLeaf() && node->getParticle() != nullptr && node->getParticle()->getId() != p->getId()) {
        if (node->getParticle()->squareDistance(*p) < (((node->getParticle()->getRadius() + p->getRadius()) *
                                                        (node->getParticle()->getRadius() + p->getRadius()))) *
                                                          softening) {
            node->getParticle()->manageCollision(*p, 0.0); //qui gestisco la collisione tra due particelle all'interno 
                                                            //dello stesso albero
        }
        //qui calcolo la forza esercitata dalla particella contenuta della foglia sul corpo p e la aggiungo nella forza
        force_qk = f.calculateForce(*p, *node->getParticle());
        p->addForce(force_qk);
    } else if (!node->isLeaf()) {
        s = node->getWidth(); //calcolo la larghezza della regione s
        approxParticle = node->createApproximateParticle();// creo la particella approssimata del nodo corrente
        d = std::sqrt(p->squareDistance(*approxParticle)); //calcolo la distanza tra il corpo e il centro di massa del nodo
        if (d == 0) {
            return;
        }
        // se il gruppo è abbastanza lontano (quindi se s/d < θ), calcolo la forza esercitata dal nodo corrente sul corpo p e la aggiungo alla forza netta di p
        if (s / d < theta) {
            force_qk = f.calculateForce(*p, *approxParticle);
            p->addForce(force_qk);
        }  //altrimenti se il gruppo non è abbastanza lontano, calcolo la forza esercitata da ciascun figlio del nodo
        else {
            for (auto& child : node->getChildren()) {
                calculateNetForceQuadtree(child, p, theta, f, dimSimulationArea, softening);
            }
        }
    }
}

/**
 * @brief Function that computes the ratio between the width of the simulation area and the distance between the particle and the center of mass.
 * 
 * @tparam Dimension Number of dimensions of the simulation
 * @param pos Position of the particle
 * @param centerMass Center of mass
 * @param width Width of the simulation area
 * @return Ratio between the width of the simulation area and the distance between the particle and the center of mass
*/
template <size_t Dimension>
double computeRatio(const std::array<double, Dimension>& pos, const std::array<double, Dimension>& centerMass,
                    double width) {
    double sum = 0.0;

    for (size_t i = 0; i < Dimension; ++i) {
        sum += std::pow(pos[i] - centerMass[i], 2);
    }

    return width / std::sqrt(sum);
    
}


/**
 * @brief Serial implementation of the Barnes-Hut algorithm: it creates the quadtree, calculates the forces acting on each particle and then updates 
 * their positions using the Verlet approximations. Finally writes the updated positions of the particles on the file after delta_t time and resets
 * the quadtree in order for it to be created again at the following iteration with the updated positions.
 * 
 * @tparam Dimension Number of dimensions of the simulation
 * @param iterationNumber Number of iterations
 * @param particles Reference to the vector of particles in the simulation
 * @param dimSimulationArea Dimension of the simulation area
 * @param softening Overhead to avoid particles overlapping and fusing together
 * @param delta_t Time step after which the simulation is updated
 * @param f Reference to the Force object responsible for calculating particle interactions.
 * @param speedUp Speedup factor
*/
template <size_t Dimension>
void serialBarnesHut(int iterationNumber, std::vector<Particle<Dimension>>* particles, int dimSimulationArea,
                     double softening, double delta_t, Force<Dimension>& f,int speedUp) {

    double theta = 0.5; //Parametro di accuratezza dell'approssimazione, da portare fuori
    std::unique_ptr<QuadtreeNode<Dimension>> quadtree;
    int numParticles = (*particles).size();
    std::FILE* file;
    std::string fileName = "../graphics/Coordinates_0.txt";
    const char* re = fileName.c_str();
    file = fopen(re, "w");
    /*
    Nel seguente if viene controllato se il file è stato aperto corettamente,
    */
    if (file == NULL) {
        std::cout << "Error opening file!" << std::endl;
        exit(1);
    }

    for (int iter = 0; iter < iterationNumber; ++iter) {
        //Vengono passati come parametri in create quadtree il puntatore di particles
        //e il doppio della dimensione dell'area di simulazione, in quanto 
        //creaQuadTree si aspetta come parametro la dimensione dell'area di simulazione, mentre
        //la funzione generateTreeParallel si aspetta come parametro la metà della dimensione 
        //dell'area di simulazione. dimSimulation deve rappresentare il raggio
        quadtree = createQuadTree(*particles, 2 * dimSimulationArea);

        for (int i = 0; i < numParticles; ++i) {
            //p viene creato come smart pointer per via del processo ricorsivo di calculateNetForceQuadtree, 
            std::shared_ptr<Particle<Dimension>> p = std::make_shared<Particle<Dimension>>((*particles)[i]);
            //Vengono calcolate le forze nel quadtree
            calculateNetForceQuadtree(quadtree, p, theta, f, dimSimulationArea, softening);
            //Vengono aggiornate le forze e le velocità delle particelle se p non è un puntatore nullo
            if (p != nullptr) (*particles)[i].addForce(p->getForce());
            //Vengono aggiornate le velocità delle particelle se p non è un puntatore nullo
            (*particles)[i].setVel(p->getVel());
        }
        //Vengono aggiornate le posizioni delle particelle e resettate le forze
        for (size_t i = 0; i < numParticles; ++i) {
            //(*particles)[i].updateAndReset(delta_t);
            (*particles)[i].velocityVerletUpdate(delta_t);
            (*particles)[i].resetForce();
        }
        if(iter%speedUp==0)
        {
            writeParticlePositionsToFile(*particles, file);
        }

        quadtree.reset();
    }
    fclose(file);
}


/**
 * * @brief Parallel implementation of the Barnes-Hut algorithm: it creates the quadtree, calculates the forces acting on each particle and then updates 
 * their positions using the Verlet approximations in parallel. Finally, resets the quadtree in order for it to be created again at the following iteration 
 * with the updated positions and writes the updated positions of the particles on the file after delta_t time in a serial way.
 * 
 * @tparam Dimension Number of dimensions of the simulation
 * @param iterationNumber Number of iterations
 * @param particles Reference to the vector of particles in the simulation
 * @param dimSimulationArea Dimension of the simulation area
 * @param softening Overhead to avoid particles overlapping and fusing together
 * @param delta_t Time step after which the simulation is updated
 * @param f Reference to the Force object responsible for calculating particle interactions.
 * @param speedUp Speedup factor
 * @param numFilesAndThreads Number of files and threads used for the simulation
 *
*/
template <size_t Dimension>
void parallelBarnesHut(int iterationNumber, std::vector<Particle<Dimension>>* particles, int dimSimulationArea,
                       double softening, double delta_t, Force<Dimension>& f,int speedUp, size_t numFilesAndThreads) {
    int num_threads = numFilesAndThreads;
    time_t start, end;
    double theta = 0.5;
    double totalTreeTime = 0.0; // Variabile per tenere traccia del tempo totale impiegato dalla funzione createQuadTree
    std::vector<std::vector<std::array<double, Dimension>>> local_forces(omp_get_max_threads(), std::vector<std::array<double, Dimension>>(particles->size()));
    //viene generato un vettore di file di output per ogni thread in modo che ogni thread possa scrivere le coordinate delle particelle in un file separato. I file si distinguono tra loro grazie al numero del thread che viene aggiunto al nome del file.
    std::vector<std::ofstream> coordinateFiles(numFilesAndThreads);
    //Viene genenerato un vettore di forza
    std::array<double,Dimension> force;
    //Inizializzo il file di output
    std::ofstream file;
    //creo un vettore di puntatori a file per ogni thread in modo che ciascun thread possa scrivere
    //le coordinate delle particelle in un file separato
    std::vector<std::FILE*> files(numFilesAndThreads);
    //initialize all files
    for (int i = 0; i < numFilesAndThreads; ++i) {
        //creo il nome del file di output per ogni thread
        std::string fileName = "../graphics/Coordinates_" + std::to_string(i) + ".txt";
        //apro il file di output per ogni thread
        const char* re = fileName.c_str();
        //nel vettore file mi associo il puntatore al file aperto per ogni thread per sola scrittura
        files[i] = fopen(re, "w");
    }
    int numParticles = (*particles).size();
    int idThread;

    //check if all files are open
    for (int i = 0; i < numFilesAndThreads; ++i) {
        if (files[i] == NULL) {
            std::cout << "Error opening file!" << std::endl;
            exit(1);
        }
    }

    //std::shared_ptr<Particle<Dimension>> p;
    std::unique_ptr<QuadtreeNode<Dimension>> quadtree;

    
    
    for (int iter = 0; iter < iterationNumber; ++iter) {
        start = time(NULL);
        quadtree = generateTreeParallel<Dimension>(particles, 2 * dimSimulationArea, num_threads);
        end = time(NULL);
        totalTreeTime = totalTreeTime + end - start;
        #pragma omp parallel shared(numParticles, particles, f, delta_t, file, softening, dimSimulationArea, num_threads, quadtree), private(idThread)
        {   
            //ogni thread deve sapere il suo id per scrivere nel file corretto
            idThread = omp_get_thread_num();
            // Calcolo delle forze per ogni particella localmente 
            #pragma omp for schedule(static, numParticles / num_threads)
            for (int i = 0; i < numParticles; ++i) {
                std::shared_ptr<Particle<Dimension>> p = std::make_shared<Particle<Dimension>>((*particles)[i]);
                calculateNetForceQuadtree(quadtree, p, theta, f, dimSimulationArea, softening);
                if (p != nullptr) {
                    (*particles)[i].addForce(p->getForce());
                    (*particles)[i].setVel(p->getVel());
                }
            }
            // Aggiornamento delle posizioni delle particelle globalmente e reset delle forze locali
            #pragma omp for schedule(static, numParticles / num_threads)
            for (auto& particle : (*particles)) {
                //particle.updateAndReset(delta_t);
                particle.velocityVerletUpdate(delta_t);
                particle.resetForce();
            }
            //fase di stampa anch'essa effettuata in parallelo visto che ciascun thread scrive in un file separato
            if(iter%speedUp==0){
                #pragma omp for schedule(static, numParticles/num_threads)
                for (int i = 0; i < numParticles; i++) {
                    fprintf(files[idThread], "%d,", (*particles)[i].getId());
                    const auto& pos = (*particles)[i].getPos();
                    for (size_t i = 0; i < Dimension; ++i) {
                        fprintf(files[idThread], "%f", pos[i]);
                        if (i < Dimension - 1) {
                            fprintf(files[idThread], ",");
                        }   
                    }
                    fprintf(files[idThread], "\n");
                }
            }
        }
        quadtree.reset();
    }
    //a fi simulazione è fondamentale che ciascun thread chiuda il file in cui ha scritto le coordinate
    #pragma omp for schedule(static, 1)
    for(int i = 0; i < numFilesAndThreads; ++i){
        fclose(files[i]);
    }
    std::cout << "Time taken by createQuadTree function: " << totalTreeTime << " seconds" << std::endl;
}



/**
 * @brief Template function that executes the NBody simulation serially for a
 *given number of iterations. For each particle,  it checks first if the
 *particle hits the boundary and manages that collision; then, it checks for
 *collision between particles and call the manageCollision function to take care
 *of it. Finally it computed the force between the particles and adds it to a
 *vector. After checking all the particles, it updates the values calculated
 *before and resets the vector; finally, it writes the updated positions of the
 *particles on the file after delta_t time.
 *
 * @tparam Dimension Number of dimensions of the simulation
 * @param it Number of iterations
 * @param particles Reference to the vector of particles
 * @param dim Dimension of the simulation area
 * @param softening Overhead to avoid particles overlapping and fusing together
 * @param delta_t Time step after which the simulation is updated
 * @param fileName Name of tile in which the function writes the coordinates of the particles computed during the simulation
 * @param file Reference to the file in which the coordinates are written
 * @param f Reference to the Force object responsible for calculating particle interactions.
 * @param speedup Speedup factor
 **/
template<size_t Dimension>
void serialSimulation(int it, std::vector<Particle<Dimension>>* particles, int dim, double softening, double delta_t, Force<Dimension>& f, int speedup){
    std::array<double,Dimension> force_qk;
    //std::ofstream file("../graphics/Coordinates_0.txt");
    std::FILE* file;
    std::string fileName = "../graphics/Coordinates_0.txt";
    const char* re = fileName.c_str();
    file = fopen(re, "w");
    /*
    Nel seguente if viene controllato se il file è stato aperto corettamente,
    altrimenti viene segnalato tramite un messaggio di errore.
    */
    if (file == NULL) {
        std::cout << "Error opening file!" << std::endl;
        exit(1);
    }
    
    for (int z = 0; z < it; ++z) {
        // std::cout<<z<<std::endl;
        for (int i = 0; i < (*particles).size(); ++i) {
            Particle<Dimension>& q = (*particles)[i];

            if (q.hitsBoundary(dim)) {
                q.manageCollision(q, dim);
            }

            for (int j = i + 1; j < (*particles).size(); j++) {
                Particle<Dimension>& k = (*particles)[j];

                if (q.squareDistance(k) <
                    (((q.getRadius() + k.getRadius()) * (q.getRadius() + k.getRadius()))) * softening) {
                    q.manageCollision(k, 0.0);//funzione da rivedere
                }
                // da tenere a mente che nel codice seriale per la massimizzazione si sfrutta la terza legge della
                //dinamica di Newton.
                force_qk = f.calculateForce(q, k);
                q.addForce(force_qk);
                for (size_t i = 0; i < Dimension; ++i) force_qk[i] = -force_qk[i];
                k.addForce(force_qk);
            }
            // Dopo aver aggiornato la posizione della particella si resetta il vettore locale
            z == it - 1 ? q.update(delta_t) : q.updateAndReset(delta_t);
        }

        if (z % speedup == 0) {
            writeParticlePositionsToFile(*particles, file);
        }
    }
    fclose(file);
}

/**
 * @brief Template function that executes the NBody simulation in parallel for a
 *given number of iterations, using OpenMP directives to parallelize the
 *simulation loop, calculating forces and updating particle positions
 *concurrently. Firstly, the function initializes the number of threads for the
 *parallel section based on the minimum between the size of the particle vector
 *or the maximum available threads. Then it starts the simulation, assigning
 *blocks to threads dynamically: collision among particles or between a particle
 *and the boundary and computation of the forces are computed concurrently and
 *then summed up. The forces are then updated and then the local vector is
 *reset; finally, the function periodically writes the updated positions inside
 *a file.
 *
 * @tparam Dimension Number of dimensions of the simulation
 * @param it Number of iterations
 * @param particles Reference to the vector of particles
 * @param dim Dimension of the simulation area
 * @param softening Overhead to avoid particles overlapping and fusing together
 * @param delta_t Time step after which the simulation is updated
 * @param fileName Name of tile in which the function writes the coordinates of the particles computed during the simulation
 * @param file Reference to the file in which the coordinates are written
 * @param f Reference to the Force object responsible for calculating particle interactions.
 * @param u Speeduo factor
 * @param numFilesAndThreads Number of files and threads used for the simulation
 **/
template<size_t Dimension>
void parallelSimulation(int it, std::vector<Particle<Dimension>>* particles, int dim, double softening, double delta_t, Force<Dimension>& f, int u, size_t numFilesAndThreads){

    int id_thread;
    int num_particles = (*particles).size();
    std::array<double,Dimension> force_qk;
    bool filesOpen = true;
    int num_threads = numFilesAndThreads;
    std::vector<std::vector<std::array<double, Dimension>>> local_forces(omp_get_max_threads(), std::vector<std::array<double, Dimension>>(particles->size()));
    std::vector<std::ofstream> coordinateFiles(numFilesAndThreads);
    std::array<double,Dimension> force;
    std::ofstream file;
    std::vector<std::FILE*> files(numFilesAndThreads);
    /*
    In questa sezione di codice vengono generati i file di output per ogni thread in modo che ogni thread 
    possa scrivere le coordinate delle particelle in un file separato. I file si distinguono tra loro
    grazie al numero del thread che viene aggiunto al nome delfile.
    */
    for (int i = 0; i < numFilesAndThreads; ++i) {
        std::string fileName = "../graphics/Coordinates_" + std::to_string(i) + ".txt";
        const char* re = fileName.c_str();
        files[i] = fopen(re, "w");
    }

    //check errori
    for (int i = 0; i < numFilesAndThreads; ++i) {
        if (files[i] == NULL) {
            std::cout << "Error opening file!" << std::endl;
            exit(1);
        }
    }

    Particle<Dimension> *q;
    Particle<Dimension> *k;
    /*
    Qui viene individuato il numero di thread da utilizzare per la sezione parallela, inoltre abbiamo definito che nella sezione parallela
    avremo anche i seguenti elementi condivisi:
    - particles: il vettore di particelle
    - f: l'oggetto Force che calcola le forze tra le particelle 
    - it: il numero di iterazioni
    - delta_t: il passo temporale
    - coordinateFiles: il vettore di file in cui scrivere le coordinate delle particelle
    - softening: il fattore di ammorbidimento
    - dim: la dimensione dell'area di simulazione
    - num_threads: il numero di thread da utilizzare
    - files: il vettore di file in cui scrivere le coordinate delle particelle
    - num_particles: il numero di particelle
    Inoltre, abbiamo definito le seguenti variabili private:
    - force: il vettore di forze tra le particelle
    - id_thread: l'id del thread corrente
    
    */

        #pragma omp parallel shared(particles, f, it, delta_t, coordinateFiles, softening, dim, num_threads, files, num_particles) private(force, id_thread, file, k, q)
    {
        id_thread = omp_get_thread_num();
    {
        for (int z = 0; z < it; ++z){
               #pragma omp for schedule(dynamic, u)
               for (int i = 0; i < num_particles; ++i) {
                    k = &(*particles)[i];


                   if(k->hitsBoundary(dim)){
                       k->manageCollision(*k, dim);
                   }

                   for(int j = i+1; j < num_particles; ++j){
                       q = &(*particles)[j];

                       if(q->squareDistance(*k) < (((q->getRadius() + k->getRadius())*(q->getRadius() + k->getRadius())))*softening){
                           q->manageCollision(*k, 0.0);
                       }

                       force = f.calculateForce(*k, *q);
                       for (int y = 0; y < Dimension; ++y) {
                           local_forces[id_thread][i][y] += force[y];
                           local_forces[id_thread][j][y] -= force[y];
                       }
                   }
               }

               #pragma omp for schedule(static, particles->size()/omp_get_num_threads())
               for (int i = 0; i < num_particles; ++i) {
                   q = &(*particles)[i];
                   q->resetForce();
                   for (int j = 0; j < num_threads; ++j) {
                       q->addForce(local_forces[j][i]);
                   }
               }

               #pragma omp for schedule(static, 1)
               for(int k = 0; k < num_threads; ++k){
                   for (int j = 0; j < num_particles; ++j) {
                       for (int y = 0; y < Dimension; ++y) {
                           local_forces[k][j][y] = 0.0;
                       }
                   }
               }

               #pragma omp for schedule(static, particles->size()/num_threads)
               for (int i = 0; i < num_particles; ++i) {
                   q = &(*particles)[i];
                   q->update(delta_t);
                }

                if(z%u==0){
                    #pragma omp for schedule(static, particles->size()/num_threads)
                    for (int i = 0; i < num_particles; i++) {
                        q = &(*particles)[i];
                        fprintf(files[id_thread], "%d,", q->getId());
                        const auto& pos = q->getPos();
                        for (size_t i = 0; i < Dimension; ++i) {
                            fprintf(files[id_thread], "%f", pos[i]);
                            if (i < Dimension - 1) {
                                fprintf(files[id_thread], ",");
                            }   
                        }
                        fprintf(files[id_thread], "\n");
                    }
                }    
        }
        #pragma omp for schedule(static, 1)
        for(int i = 0; i < numFilesAndThreads; ++i){
            fclose(files[i]);
        }
    }
    }
}

/**
 * @brief Template function that wraps main function for the 2D simulation:
 *calls the function that generates the particles, the one that prints the
 *initial states on the file and then calls the function which starts the
 *simulation chosen by the user.
 *
 * @tparam Dimension Number of dimensions of the simulation
 * @param simType Simulation type: 0 for serial, 1 for parallel, 2 for serial Barnes Hut, 3 for parallel Barnes Hut
 * @param forceType Type of the force: g for gravitational force, c for coulomb force
 * @param delta_t Time step after which the simulation is updated
 * @param dimSimulationArea Dimension of the simulation area
 * @param iterationNumber Number of iterations
 * @param numParticles Number of particles
 * @param mass Property of the particles: mass for the gravitational particles, charge for the coulomb particles
 * @param maxVel Maximum velocity of the particles
 * @param maxRadius Maximum radius of the particles
 * @param softening Overhead to avoid particles overlapping and fusing together
 * @param fileName Name of tile in which the function writes the coordinates of the particles computed during the simulation
 *
 **/
template <size_t Dimension>
void main2DSimulation(int forceType, int simType, double delta_t, int dimSimulationArea, int iterationNumber,
                      int numParticles, int mass, int maxVel, int maxRadius, double softening, std::string fileName,
                      int speedUp) {
    time_t start, end;
    std::vector<Particle<Dimension>> particles;
    size_t numFilesAndThreads;
    Force<Dimension>* f;
    if (forceType == 1)
        f = new CoulombForce<Dimension>();
    else
        f = new GravitationalForce<Dimension>();
    /*
        Sezione geerazione particelle: la fuznione generate RandomParticles genera un numero di particelle pari a 
        numParticles, con massa pari a mass, velocità massima pari a maxVel e raggio massimo pari a maxRadius.
        La funzione restituisce un vettore di particelle che viene salvato nella variabile particles.
    */
    start = time(NULL);
    particles = generateRandomParticles<Dimension>(numParticles, dimSimulationArea, (forceType) ? -mass : 1, mass,
                                                   maxVel, 1, maxRadius, forceType);

    end = time(NULL);
    std::cout << "Time taken by generateRandomParticles function: " << end - start << " seconds" << std::endl;

    if(!simType) numFilesAndThreads = 1;
    else{
        particles.size() < omp_get_max_threads()? omp_set_num_threads(particles.size()):omp_set_num_threads(omp_get_max_threads());  
        numFilesAndThreads = omp_get_max_threads();
    }
    printInitialStateOnFile(&particles, dimSimulationArea, fileName, iterationNumber, speedUp, numFilesAndThreads);
    //DA CORREGGERE PER FINI SIMULAZIONE ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (simType == 0) {
        start = time(NULL);
        serialSimulation<Dimension>(iterationNumber, &particles, dimSimulationArea, softening, delta_t, *f, speedUp);
        end = time(NULL);
        std::cout << "Time taken by serial simulation: " << end - start << " seconds" << std::endl;
    } else if (simType == 1) {
        start = time(NULL);
        parallelSimulation<Dimension>(iterationNumber, &particles, dimSimulationArea, softening, delta_t, *f, speedUp, numFilesAndThreads);
        end = time(NULL);
        std::cout << "Time taken by parallel simulation: " << end - start << " seconds" << std::endl;
    } else if (simType == 2) {
        start = time(NULL);
        serialBarnesHut<Dimension>(iterationNumber, &particles, dimSimulationArea, softening, delta_t, *f, speedUp);
        end = time(NULL);
        std::cout << "Time taken by serial Barnes Hut: " << end - start << " seconds" << std::endl;
    } else if (simType == 3) {
        start = time(NULL);
        parallelBarnesHut<Dimension>(iterationNumber, &particles, dimSimulationArea, softening, delta_t, *f, speedUp, numFilesAndThreads);
        end = time(NULL);
        std::cout << "Time taken by parallel Barnes Hut: " << end - start << " seconds" << std::endl;
    }
}

/**
 * @brief Template function that wraps main function for the 3D simulation:
 *calls the function that generates the particles, the one that prints the
 *initial states on the file and then calls the function which starts the
 *simulation chosen by the user.
 *
 * @tparam Dimension Number of dimensions of the simulation
 * @param simType Simulation type: 0 for serial, 1 for parallel
 * @param forceType Type of the force: g for gravitational force, c for coulomb force
 * @param delta_t Time step after which the simulation is updated
 * @param dimSimulationArea Dimension of the simulation area
 * @param iterationNumber Number of iterations
 * @param numParticles Number of particles
 * @param mass Property of the particles: mass for the gravitational particles, charge for the coulomb particles
 * @param maxVel Maximum velocity of the particles
 * @param maxRadius Maximum radius of the particles
 * @param softening Overhead to avoid particles overlapping and fusing together
 * @param fileName Name of tile in which the function writes the coordinates of the particles computed during the simulation
 **/
template <size_t Dimension>
void main3DSimulation(int forceType, int symType, double delta_t, int dimSimulationArea, int iterationNumber,
                      int numParticles, int mass, int maxVel, int maxRadius, int softening, std::string fileName,
                      int speedUp) {
    time_t start, end;
    std::vector<Particle<Dimension>> particles;
    Force<Dimension>* f; //this pointer is used to point to the force object that will be used in the simulation.
    size_t numFilesAndThreads;
    if (forceType == 1)
        f = new CoulombForce<Dimension>();
    else
        f = new GravitationalForce<Dimension>();

    start = time(NULL);
    //da rivedere la definizione in base al metodo da ristrutturare
    particles = generateRandomParticles<Dimension>(numParticles, dimSimulationArea, (forceType) ? -mass : 1, mass, maxVel, 1, maxRadius, forceType);
    end = time(NULL);
    std::cout << "Time taken by generateRandomParticles function: " << end - start << " seconds" << std::endl;
    //In questa sezione, a seconda che la simulazione sia seriale o parallela, viene impostato il numero
    //di threads utilizzati e il numero di files in cui vengono scritte le coordinate delle particellle
    if(!symType) numFilesAndThreads = 1;
    else{
        particles.size() < omp_get_max_threads()? omp_set_num_threads(particles.size()):omp_set_num_threads(omp_get_max_threads());  
        numFilesAndThreads = omp_get_max_threads();
    }
    //dopo la generazione delle particelle viene stampato lo stato iniziale delle particelle in un file di testo,
    //che contiene le informazioni sulle particelle, come la loro posizione, velocità e massa.
    //-->da rivedere la funzione printInitialStateOnFile, che non è presente nel codice
    printInitialStateOnFile(&particles, dimSimulationArea, fileName, iterationNumber, speedUp, numFilesAndThreads);

    if (symType == 0) {
        start = time(NULL);
        serialSimulation<Dimension>(iterationNumber, &particles, dimSimulationArea, softening, delta_t, *f, speedUp);
        end = time(NULL);
        std::cout << "Time taken by serial simulation: " << end - start << "seconds" << std::endl;

    } else if (symType == 1) {
        start = time(NULL);
        parallelSimulation<Dimension>(iterationNumber, &particles, dimSimulationArea, softening, delta_t, *f, speedUp, numFilesAndThreads);
        end = time(NULL);
        std::cout << "Time taken by parallel simulation: " << end - start << " seconds" << std::endl;
    }

}

/**
 * @brief Helper function for the user that is called when the user runs the program with 
 * the -h flag. It prints the list of all the parameters that can be changed by the user and their description.
*/
void showHelp() {
    std::cout << "Change the following parameters if you don't want to run the default simualtion: "<< std::endl;
    std::cout << "      -h : prints helper " << std::endl;
    std::cout << "      -dim <int> : number of dimension of the simulation (2D,3D) " << std::endl;
    std::cout << "      -simT <int> : simulation type (0 for serial, 1 for parallel) " << std::endl;
    std::cout << "      -force <int> : type of the force of the simulation " << std::endl;
    std::cout << "      -delta <double>: time step of the simulation " << std::endl;
    std::cout << "      -simA <int> : dimension of the simulation area " << std::endl;
    std::cout << "      -it <int> : iteration number " << std::endl;
    std::cout << "      -numP <int> : number of particles " << std::endl;
    std::cout << "      -maxPr <int> : maximum property of the particle(mass, charge) " << std::endl;
    std::cout << "      -maxVel <int> : maximum velocity of the particles " << std::endl;
    std::cout << "      -maxR <int> : maximum radius of the particles " << std::endl;
    std::cout << "      -soft <double> : softener of the particles " << std::endl;
    std::cout << "      -spUp <int> : speedup of the simulation  " << std::endl;
    std::cout << "      -file <std::string> : file in which the output is written  " << std::endl;
}

/**
 * @brief Main function which asks the users for the simulation type, the
 * dimensions and the force type and then runs the simulation accordingly
 */

int main(int argc, char** argv) {
    //Here I am setting the default values for the simulation parameters, 
    //which can be changed by the user through command line arguments. 
    int dim = 2;
    int simType = 1;
    int forceType = 0;
    double delta_t = 0.01;
    int dimSimulationArea = 500;
    int iterationNumber = 1000;
    int numParticles = 50;
    int mass = 1000;
    int maxVel = 5;
    int maxRadius = 10;
    double softening = 0.7;
    int speedUp = 1;
    std::string fileName = "../graphics/Info.txt";

    if (argc < 2) {
        char a;
        std::cout << "Enter 'd' to run the default simulation: " << std::endl;
        std::cin >> a;
        if (a == 'd')
            dim = 2;
        else {
            showHelp();
            return 1;
        }
    }
    /*
        In this section of the code the for loop iterates through the command line arguments
        passed to the program and checks for specific flags that indicate the simultation 
        parameter, such as the dimensio of simulation, the type of simulation (serial or parallel),
        the type of force( gravitational or coulomb),the timestep, the dimnsion of the simulation area
        the number of iterations, number of particles, the maximum prorpety of particles 
        and the maximum velocity and radius of the particles.
        If a flag is found the code chcks if it is valid and if it is, it assigns 
        the corresponding value  to the variable . IF the flag is not valid or missing, it returns an error message
        and exits the program.
    */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-dim") == 0) {
            if (++i < argc) {
                dim = atoi(argv[i]);
                if (dim != 2 && dim != 3) {
                    std::cout << "No feasible dimension." << std::endl;
                    return 1;
                }
            } else {
                std::cout << " Error: flag -dim requires values 2 or 3 to work. " << std::endl;
                return 1;
            }
        }
        /*
        if strcmp is meant to compare the command line argument with the string "-dim". If they are equal,
        it means that the user has specified the dimension of the simultation, and the code checks if the next argument 
        is valid and assigns it to the variable dim. If the argument is not valid or missing, it terminates
        the program.
        */

        if (strcmp(argv[i], "-simT") == 0) {
            if (++i < argc) {
                simType = atoi(argv[i]);
                if (simType != 0 && simType != 1 && simType != 2 && simType != 3) {
                    std::cout << "No feasible simulation." << std::endl;
                    return 1;
                }
            } else {
                std::cout << " Error: flag -simT requires values 0 for serial "
                             "or 1 for parallel to work. "
                          << std::endl;
                return 1;
            }
        }
        /*
          I check if the force term is equal to "-force" and if it is, I check if the next argument 
          is valid and assign it to the variable forceType. If the argument is not valid or missing, 
          it terminates the program.
        */
        if (strcmp(argv[i], "-force") == 0) {
            if (++i < argc) {
                forceType = atoi(argv[i]);
                if (forceType != 0 && forceType != 1) {
                    std::cout << "No feasible force." << std::endl;
                    return 1;
                }
            } else {
                std::cout << " Error: flag -force requires values 0 for "
                             "gravitational force or 1 for Coulomb force to work. "
                          << std::endl;
                return 1;
            }
        }
        /*
         In this section I am checking if the command line argument is equal to "-delta" and if it is, 
         I check if the next argument is valid and assign it 
         to the variable delta_t. If the argument is not valid or missing, it terminates the program.
        */

        if (strcmp(argv[i], "-delta") == 0) {
            if (++i < argc) {
                double delta = std::__cxx11::stof(argv[i]);
                if (delta < 0) {
                    std::cout << "No feasible delta t." << std::endl;
                    return 1;
                }
                delta_t = delta;
            } else {
                std::cout << " Error: flag -delta requires a positive value to work. " << std::endl;
                return 1;
            }
        }
        /*
        In this section I am checking if the command line argument is equal to "-simA" and if it is, 
         I check if the next argument is valid and assign it to the variable dimSimulationArea. If the 
         argument is not valid or missing, it terminates the program.
        */

        if (strcmp(argv[i], "-simA") == 0) {
            if (++i < argc) {
                int simArea = atoi(argv[i]);
                if (simArea < 0) {
                    std::cout << "No feasible simulation area." << std::endl;
                    return 1;
                }
                dimSimulationArea = simArea;
            } else {
                std::cout << " Error: flag -simA requires a positive value to work. " << std::endl;
                return 1;
            }
        }
        /*
        In this section I am checking if the command line argument is equal to "-it" and if it is, 
         I check if the next argument is valid and assign it to the variable iterationNumber. If the 
         argument is not valid or missing, it terminates the program.
        */

        if (strcmp(argv[i], "-it") == 0) {
            if (++i < argc) {
                int it = atoi(argv[i]);
                if (it < 0) {
                    std::cout << "No feasible number of iterations." << std::endl;
                    return 1;
                }
                iterationNumber = it;
            } else {
                std::cout << " Error: flag -it requires a positive value to work. " << std::endl;
                return 1;
            }
        }
        /*
            I check if the command line argument is equal to "-numP" and if it is, I check if the next argument
            is valid and assign it to the variable numParticles. If the argument is not valid or missing,
            it terminates the program.
        */

        if (strcmp(argv[i], "-numP") == 0) {
            if (++i < argc) {
                int numP = atoi(argv[i]);
                if (numP < 0) {
                    std::cout << "No feasible number of particles." << std::endl;
                    return 1;
                }
                numParticles = numP;
            } else {
                std::cout << " Error: flag -numP requires a positive value to work. " << std::endl;
                return 1;
            }
        }

        /*
            In this section I am checking if the command line argument is equal to "-maxPr" and if it is, 
             I check if the next argument is valid and assign it to the variable mass. If the 
             argument is not valid or missing, it terminates the program.
        */
        if (strcmp(argv[i], "-maxPr") == 0) {
            if (++i < argc) {
                int maxPr = atoi(argv[i]);
                if (maxPr < 0) {
                    std::cout << "No feasible value of maximum property." << std::endl;
                    return 1;
                }
                mass = maxPr;
            } else {
                std::cout << " Error: flag -maxPr requires a positive value to work. " << std::endl;
                return 1;
            }
        }

        /*
            In this section I am checking if the command line argument is equal to "-maxVel" and if it is, 
             I check if the next argument is valid and assign it to the variable maxVel. If the 
             argument is not valid or missing, it terminates the program.
        */
        if (strcmp(argv[i], "-maxVel") == 0) {
            if (++i < argc) {
                int maxV = atoi(argv[i]);
                if (maxV < 0) {
                    std::cout << "No feasible value of radius of the particles." << std::endl;
                    return 1;
                }
                maxVel = maxV;
            } else {
                std::cout << " Error: flag -maxVEl requires a positive value "
                             "to work. "
                          << std::endl;
                return 1;
            }
        }
        /*
            In this section I am checking if the command line argument is equal to "-maxR" and if it is, 
             I check if the next argument is valid and assign it to the variable maxRadius. If the 
             argument is not valid or missing, it terminates the program.
        */
        if (strcmp(argv[i], "-maxR") == 0) {
            if (++i < argc) {
                int maxR = atoi(argv[i]);
                if (maxR < 0) {
                    std::cout << "No feasible value of radius of the particles." << std::endl;
                    return 1;
                }
                maxRadius = maxR;
            } else {
                std::cout << " Error: flag -maxR requires a positive value to work. " << std::endl;
                return 1;
            }
        }
        /*
            In this section I am checking if the command line argument is equal to "-soft" and if it is, 
             I check if the next argument is valid and assign it to the variable softening. If the 
             argument is not valid or missing, it terminates the program.
        */

        if (strcmp(argv[i], "-soft") == 0) {
            if (++i < argc) {
                double soft = std::__cxx11::stof(argv[i]);
                if (soft < 0) {
                    std::cout << "No feasible value of softening." << std::endl;
                    return 1;
                }
                softening = soft;
            } else {
                std::cout << " Error: flag -soft requires a positive value to work. " << std::endl;
                return 1;
            }
        }
        /*
            In this section I am checking if the command line argument is equal to "-spUp" and if it is, 
             I check if the next argument is valid and assign it to the variable speedUp. If the 
             argument is not valid or missing, it terminates the program.
        */

        if (strcmp(argv[i], "-spUp") == 0) {
            if (++i < argc) {
                int spUp = atoi(argv[i]);
                if (spUp < 0) {
                    std::cout << "No feasible value of speedup." << std::endl;
                    return 1;
                }
                speedUp = spUp;
            } else {
                std::cout << " Error: flag -spUp requires a positive value to work. " << std::endl;
                return 1;
            }
        }
        /*
            In this section I am checking if the command line argument is equal to "-file" and if it is, 
             I check if the next argument is valid and assign it to the variable fileName. If the 
             argument is not valid or missing, it terminates the program.
        */

        if (strcmp(argv[i], "-file") == 0) {
            if (++i < argc) {
                std::string file = argv[i];
                fileName = file;
            } else {
                std::cout << " Error: flag -file requires a valid file name to work. " << std::endl;
                return 1;
            }
        }
        /*
            In this section I am checking if the command line argument is equal to "-h" and if it is,
            I call the showHelp() function to display the help message and then return 0 to exit the program.
        */
        if (strcmp(argv[i], "-h") == 0) {
            showHelp();
            return 0;
        }
    }

    if (dim == 2) {
        main2DSimulation<2>(forceType, simType, delta_t, dimSimulationArea, iterationNumber, numParticles, mass, maxVel,
                            maxRadius, softening, fileName, speedUp);
    } else if (dim == 3) {
        main3DSimulation<3>(forceType, simType, delta_t, dimSimulationArea,
        iterationNumber, numParticles, mass, maxVel, maxRadius, softening,
        fileName, speedUp);
    }

    return 0;
}
