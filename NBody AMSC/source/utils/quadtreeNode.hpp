#ifndef QUADTREE_NODE_HPP
#define QUADTREE_NODE_HPP

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "particle.hpp"

using namespace std;

template <size_t Dimension>
class Particle;

/**
 * @brief Node of the two-dimensional quadtree used by the Barnes-Hut algorithm.
 *
 * Each node represents a square region of the simulation domain. A leaf stores
 * one particle directly and may store additional particles when the maximum
 * depth is reached. An internal node owns four child nodes and stores aggregate
 * information used to approximate distant groups of particles.
 *
 * @tparam Dimension Number of coordinates stored by each particle. The current
 * implementation uses the first two coordinates to build four quadrants.
 *
 
 */
template <size_t Dimension>
class QuadtreeNode {
   public:
     /**
      * @brief Creates an empty leaf node.
      * @param x X coordinate of the node center
      * @param y Y coordinate of the node center
      * @param w Width of the square region represented by the node
      * @param maxDepthValue Maximum recursion depth allowed for particle insertion
      */
    QuadtreeNode(double x, double y, double w, int maxDepthValue = 10)
        : width(w),
          leaf(true),
          totalCenter(std::array<double, Dimension>()),
          center({x, y}),
          totalProperty(0.0),
          count(0),
          maxDepth(maxDepthValue) {
    }

    /** @brief Returns the geometric center of this node. */
    const std::array<double, Dimension>& getCenter() const { return center; }

    /** @brief Returns the width of the square region represented by this node. */
    double getWidth() const { return width; }

    /**
     * @brief Checks whether a position belongs to this node's square region.
     * @param position Position to test
     * @return true if the position is inside the region, otherwise false
     */
    bool contains(const std::array<double, Dimension>& position) const {
        // Il nodo e' un quadrato: la distanza dal centro e' la stessa sui due assi.
        const double halfWidth = width / 2.0;
        return position[0] >= center[0] - halfWidth && position[0] <= center[0] + halfWidth &&
               position[1] >= center[1] - halfWidth && position[1] <= center[1] + halfWidth;
    }

    /** @brief Returns whether this node currently has no children. */
    bool isLeaf() const { return leaf; }

    /** @brief Returns the particle stored directly in this leaf, if any.
     *  Il puntatore punta dentro il vector originale delle particelle: il nodo
     *  non possiede la particella, la referenzia soltanto. */
    const Particle<Dimension>* getParticle() const { return particle; }

    /** @brief Returns particles stored as overflow at the depth limit.
     *  Stesso contratto di getParticle(): puntatori non posseduti. */
    const std::vector<const Particle<Dimension>*>& getAdditionalParticles() const {
        return additionalParticles;
    }

    /** @brief Returns the aggregate center weighted by particle properties. */
    const std::array<double, Dimension>& getTotalCenter() const { return totalCenter; }

    /** @brief Returns the sum of particle properties contained in this node. */
    double getTotalProperty() const { return totalProperty; }

    /** @brief Backward-compatible alias for code that expects a total mass. */
    double getTotalMass() const { return totalProperty; }

    /** @brief Returns the number of particles represented by this node. */
    int getCount() const { return count; }

    /** @brief Returns the four child nodes of this node. */
    const std::array<std::unique_ptr<QuadtreeNode<Dimension>>, 4>& getChildren() const {
        return children;
    }

    /**
     * @brief Inserts an already-built subtree into this node.
     *
     * The subtree is placed according to its center. If the selected child is
     * occupied, insertion continues recursively instead of replacing it.
     *
     * @param subTreeRoot Subtree transferred to this node
     * @param depth Current recursion depth
     */
    void insertNode(std::unique_ptr<QuadtreeNode<Dimension>> subTreeRoot, int depth = 0) {
        // Un unique_ptr nullo non rappresenta un sottoalbero valido.
        if (!subTreeRoot) {
            cerr << "Error: the node being inserted is null" << endl;
            return;
        }

        // Il nodo padre incorpora i dati aggregati del sottoalbero ricevuto.
        updateAttributes(subTreeRoot->createApproximateParticle(), subTreeRoot->getCount());

        // Se il nodo era una foglia, la trasformazione in nodo interno crea i quattro figli.
        if (leaf) {
            split();
        }

        // Il centro del sottoalbero determina il quadrante nel quale inserirlo.
        int index = getQuadrantIndex(subTreeRoot->getCenter());
        if (index == -1 || children[index] == nullptr) {
            cerr << "Error: invalid child index for the node to be inserted" << endl;
            return;
        }

        // getParticle() non basta: un nodo interno puo' non avere una particella diretta
        // ma contenere comunque particelle nei propri discendenti.
        if (children[index]->getCount() == 0) {
            children[index] = std::move(subTreeRoot);
        } else {
            // Il quadrante e' occupato: si scende ricorsivamente senza perdere il sottoalbero.
            children[index]->insertNode(std::move(subTreeRoot), depth + 1);
        }
    }

    /**
     * @brief Splits a leaf into four equally sized child nodes.
     *
     * A node whose width is already too small is left unchanged. The width check
     * must happen before changing the leaf flag, otherwise the node could become
     * internal without receiving valid children.
     */
    void split() {
        // Prima verifico la larghezza per non lasciare un nodo interno senza figli.
        if (width <= std::numeric_limits<double>::epsilon()) return;

        // Da questo punto il nodo puo' essere rappresentato dai quattro figli.
        leaf = false;
        const double halfWidth = width / 2.0;
        const double quarterWidth = width / 4.0;
        const double x = center[0];
        const double y = center[1];

        children[0] = std::make_unique<QuadtreeNode<Dimension>>(
            x - quarterWidth, y - quarterWidth, halfWidth);
        children[1] = std::make_unique<QuadtreeNode<Dimension>>(
            x + quarterWidth, y - quarterWidth, halfWidth);
        children[2] = std::make_unique<QuadtreeNode<Dimension>>(
            x - quarterWidth, y + quarterWidth, halfWidth);
        children[3] = std::make_unique<QuadtreeNode<Dimension>>(
            x + quarterWidth, y + quarterWidth, halfWidth);
    }

    /**
     * @brief Returns the child index corresponding to a position.
     * @param pos Position whose quadrant must be found
     * @return Index in the range [0, 3], or -1 for an invalid position
     */
    int getQuadrantIndex(const std::array<double, Dimension>& pos) {
        // Il confronto con il centro decide se la posizione e' a sinistra o a destra.
        const bool isLeft = pos[0] < center[0];
        // Il secondo confronto decide se la posizione e' sopra o sotto il centro.
        const bool isTop = pos[1] < center[1];

        if (isLeft && isTop) return 0;
        if (!isLeft && isTop) return 1;
        if (isLeft && !isTop) return 2;
        if (!isLeft && !isTop) return 3;
        return -1;
    }

    /**
     * @brief Inserts a particle and updates the node aggregates.
     *
     * The first particle can be stored directly in a leaf. When a second
     * particle arrives, the leaf is split and both particles are redistributed.
     * If the maximum depth or the minimum width is reached, the particle is kept
     * in the overflow collection instead.
     *
     * @param p Particle to insert
     * @param depth Current recursion depth
     */
    void insert(const Particle<Dimension>* p, int depth = 0) {
        // Un puntatore nullo non puo' aggiornare ne' la struttura ne' gli aggregati.
        if (!p) {
            cerr << "Error: the particle being inserted is null" << endl;
            return;
        }

        // Ogni particella contribuisce alle proprieta' aggregate del nodo corrente.
        updateAttributes(p);

        // Una foglia vuota conserva direttamente il primo elemento.
        if (leaf && particle == nullptr) {
            particle = p;
            return;
        }

        // Se non e' possibile dividere ulteriormente, si usa la lista di overflow.
        if (leaf && particle != nullptr &&
            (depth >= maxDepth || width <= std::numeric_limits<double>::epsilon())) {
            additionalParticles.push_back(p);
            return;
        }

        // La foglia contiene gia' una particella: la si sposta nel figlio corretto.
        if (leaf && particle != nullptr) {
            split();
            int index = getQuadrantIndex(particle->getPos());
            if (index != -1 && children[index] != nullptr) {
                children[index]->insert(particle, depth + 1);
            } else {
                cerr << "Error: quadrant index invalid while repositioning" << endl;
            }
            particle = nullptr;
        }

        // Anche la nuova particella viene inoltrata al figlio corrispondente.
        int index = getQuadrantIndex(p->getPos());
        if (index != -1 && children[index] != nullptr) {
            children[index]->insert(p, depth + 1);
        } else {
            cerr << "Error: invalid child index for the new particle" << endl;
        }
    }

    /**
     * @brief Updates aggregates using a raw, non-owning particle pointer.
     * @param newParticle Particle to include in the aggregates
     * @param numParticles Number of represented particles
     */
    void updateAttributes(const Particle<Dimension>* newParticle, int numParticles = 1) {
        // Il sovraccarico per riferimento evita di duplicare la formula di aggiornamento.
        if (newParticle) {
            updateAttributes(*newParticle, numParticles);
        }
    }

    /**
     * @brief Updates the total property, particle count and weighted center.
     *
     * The aggregate property can be a mass or another signed particle property,
     * such as electric charge. When the new total is zero or numerically close to
     * zero, the weighted center is reset instead of dividing by zero.
     *
     * @param newParticle Particle or aggregate particle to include
     * @param numParticles Number of represented particles
     */
    void updateAttributes(const Particle<Dimension>& newParticle, int numParticles = 1) {
        // Salvo il totale precedente per poter aggiornare il centro in modo incrementale.
        const double oldTotalProperty = totalProperty;
        // Il totale puo' essere una massa oppure una carica, a seconda del modello.
        totalProperty += newParticle.getProperty();
        // Un sottoalbero puo' rappresentare piu' particelle, non solo una.
        count += numParticles;

        // Con proprieta' totali nulle il centro pesato non e' definito.
        if (std::abs(totalProperty) <= std::numeric_limits<double>::epsilon()) {
            totalCenter.fill(0.0);
            return;
        }

        for (size_t i = 0; i < Dimension; ++i) {
            // Ricostruisco la somma pesata precedente e aggiungo il nuovo contributo.
            const double weightedPosition = totalCenter[i] * oldTotalProperty +
                                             newParticle.getPos()[i] * newParticle.getProperty();
            // Divido per il nuovo totale per ottenere il centro aggregato aggiornato.
            totalCenter[i] = weightedPosition / totalProperty;
        }
    }

    /**
     * @brief Prints this node and recursively prints all its descendants.
     * @param depth Indentation depth used for readable output
     */
    void printTree(int depth = 0) const {
        // L'indentazione visualizza il livello del nodo nella gerarchia.
        cout << string(depth * 4, ' ') << "Node at depth " << depth << ": [" << center[0] << ", "
             << center[1] << ", " << width << "], Property: " << totalProperty
             << ", Particles: " << count << "," << (leaf ? " Leaf node\n" : " Internal node\n");

        // Le foglie possono contenere una particella diretta o elementi di overflow.
        if (leaf) {
            if (particle != nullptr) {
                cout << string((depth + 1) * 4, ' ') << "Particle: " << particle->getId() << ") "
                     << particle->getPos()[0] << ", " << particle->getPos()[1]
                     << ". property: " << particle->getProperty() << "\n";
            }
        } else {
            // Nei nodi interni seguo ricorsivamente solo i figli realmente presenti.
            for (const auto& child : children) {
                if (child != nullptr) {
                    child->printTree(depth + 1);
                }
            }
        }
    }

    /**
     * @brief Creates a particle representing all particles in this node.
     *
     * The returned particle uses the aggregate property and weighted center,
     * and is intended for distant-node force approximation in Barnes-Hut.
     *
     * @return Aggregate particle with identifier -1
     */
    Particle<Dimension> createApproximateParticle() const {
        // La velocita' non serve per l'approssimazione della forza e viene inizializzata a zero.
        std::array<double, Dimension> pos{};
        std::array<double, Dimension> vel{};
        for (size_t i = 0; i < Dimension; ++i) {
            // Il centro della particella approssimata coincide con il centro aggregato.
            pos[i] = totalCenter[i];
        }
        return Particle<Dimension>(-1, totalProperty, pos, vel, 1.0, false);
    }

   private:
    double width;
    bool leaf;
    
    const Particle<Dimension>* particle = nullptr;
    std::vector<const Particle<Dimension>*> additionalParticles;
    std::array<std::unique_ptr<QuadtreeNode<Dimension>>, 4> children;
    std::array<double, Dimension> totalCenter;
    std::array<double, Dimension> center;
    double totalProperty;
    int count;
    int maxDepth;
};

#endif  // QUADTREE_NODE_H
