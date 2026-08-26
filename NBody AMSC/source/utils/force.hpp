#ifndef FORCE_H
#define FORCE_H

#include <array>
#include <cmath>

template<size_t Dimension>
class Particle;

/**
 * @brief Force class for particle interactions.
 * @tparam Dimension Number of dimensions of the simulation 
 */
template<size_t Dimension>
class Force{
    public:
        //distruttore di default della classe Force    
        virtual ~Force() = default;

        /**
         * @brief Pure virtual function for calculating the force between two particles.
         * @param p1 First particle involved in the force calculation.
         * @param p2 Second particle involved in the force calculation.
         * @return Array representing the force vector between particles.
         */
        virtual std::array<double,Dimension> calculateForce(const Particle<Dimension> &p1, const Particle<Dimension> &p2) const = 0;

    protected:
        //calcolo la distanza tra due particelle e ritorno il vettore di spostamento
        static std::array<double, Dimension> calculateDisplacement(
            const Particle<Dimension> &first, const Particle<Dimension> &second) {
            std::array<double, Dimension> displacement{};
            for (size_t i = 0; i < Dimension; ++i) {
                displacement[i] = second.getPos()[i] - first.getPos()[i];
            }
            return displacement;
        }
        //calcolo distanza quadratica
        static double calculateDistanceSquared(
            const std::array<double, Dimension> &displacement) {
            double distanceSquared = 0.0;
            for (size_t i = 0; i < Dimension; ++i) {
                distanceSquared += displacement[i] * displacement[i];
            }
            constexpr double minimumDistanceSquared = 1e-24;
            return distanceSquared < minimumDistanceSquared
                       ? minimumDistanceSquared
                       : distanceSquared;
        }

        static double distanceCubed(double distanceSquared) {
            return distanceSquared * std::sqrt(distanceSquared);
        }

        static std::array<double, Dimension> assembleForce(
                                                            const std::array<double, Dimension> &displacement,
                                                            double factor) {
            std::array<double, Dimension> force{};
            for (size_t i = 0; i < Dimension; ++i) {
                force[i] = factor * displacement[i];
            }
            return force;
        }
    };


/**
 * @brief GravitationalForce class models the gravitational force and inherits from Force.
 * @tparam Dimension Number of dimensions of the simulation 
 */
template<size_t Dimension>
class GravitationalForce : public Force<Dimension>{
    public:
        /**
         * @brief Override of function for calculating the force between two particles in order to adapt it to the gravitational force.
         * @param p1 First particle involved in the force calculation.
         * @param p2 Second particle involved in the force calculation.
         * @return Array representing the force vector between particles.
         */
        std::array<double,Dimension> calculateForce(const Particle<Dimension> &k, const Particle<Dimension> &q) const override{
            const auto displacement = this->calculateDisplacement(k, q);
            const double distanceSquared = this->calculateDistanceSquared(displacement);
            const double factor = G * q.getProperty() * k.getProperty() /
                                  this->distanceCubed(distanceSquared);
            return this->assembleForce(displacement, factor);
        }
    private:
        static constexpr double G = 6.674e-11;
};

/**
 * @brief CoulombForce class models the Coulomb force and inherits from Force.
 * @tparam Dimension Number of dimensions of the simulation 
 */
template<size_t Dimension>
class CoulombForce : public Force<Dimension>{
    public:
        /**
         * @brief Override of function for calculating the force between two particles in order to adapt it to the Coulomb force.
         * @param k First particle involved in the force calculation.
         * @param q Second particle involved in the force calculation.
         * @return Array representing the force vector between particles.
         */
        std::array<double,Dimension> calculateForce(const Particle<Dimension> &k, const Particle<Dimension> &q) const override{
            const auto displacement = this->calculateDisplacement(k, q);
            const double distanceSquared = this->calculateDistanceSquared(displacement);
            const double factor = -K * q.getProperty() * k.getProperty() /
                                  this->distanceCubed(distanceSquared);
            return this->assembleForce(displacement, factor);
        }
    private:
        static constexpr double K = 8.987e-09;
};



/**
 * @brief RepulsiveForce class models the repulsive force and inherits from Force.
 * @tparam Dimension Number of dimensions of the simulation 
 */
template<size_t Dimension>
class RepulsiveForce : public Force<Dimension>{
    public: 
        /**
         * @brief Override of function for calculating the force between two particles in order to adapt it to the repulsive force.
         * @param k First particle involved in the force calculation.
         * @param q Second particle involved in the force calculation.
         * @return Array representing the force vector between particles.
         */
        std::array<double, Dimension> calculateForce(const Particle<Dimension> &k, const Particle<Dimension> &q) const override{
            const auto displacement = this->calculateDisplacement(k, q);
            const double distanceSquared = this->calculateDistanceSquared(displacement);
            const double factor = -kp / (distanceSquared * distanceSquared);
            return this->assembleForce(displacement, factor);
        }
    private:
        static constexpr double kp = 1.0;
};

/**
 * @brief CustomForce class models a custom force similar to the gravitational force and inherits from Force. 
 * For test purposes only
 * @tparam Dimension Number of dimensions of the simulation 
 */
template<size_t Dimension>
class CustomForce : public Force<Dimension>{
    public:
        CustomForce(double G) : G(G) {}
        std::array<double,Dimension> calculateForce(const Particle<Dimension> &k, const Particle<Dimension> &q) const override{
            const auto displacement = this->calculateDisplacement(k, q);
            const double distanceSquared = this->calculateDistanceSquared(displacement);
            const double factor = G * q.getProperty() * k.getProperty() /
                                  this->distanceCubed(distanceSquared);
            return this->assembleForce(displacement, factor);
        }
    private:
        double const G; 
};

#endif
