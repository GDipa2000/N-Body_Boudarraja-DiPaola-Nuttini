#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <iostream>
#include <array>
#include <cmath>


template<size_t Dimension>
class Force;

/**
 * @brief Particle class that models particles
 * @tparam Dimension Number of dimensions of the simulation 
 */
template<size_t Dimension>
class Particle {
    public:
        /**
        * @brief Constructor that initializes the mass, position, and velocity of the particle
        * @param id Id of the particle
        * @param p Property of particle: mass for gravitational force, charge for Coulomb force
        * @param pos Array of positions of the particle
        * @param v Array of velocities of the particle
        * @param radius Radius of the particle
        * @param type Type of the particle: 0 for gravitational, 1 for Coulomb 
        * */
        Particle(int id, double p, const std::array<double, Dimension> &pos,
                 const std::array<double, Dimension> &v, double radius, bool type)
            : id(id), property(p), pos(pos), vel(v), force{}, accel{}, type(type), radius(radius) {}

        /**
         * @brief Setter method for velocity of the particle
         * @param v Array of velocities of the particle
         * 
        */
        void setVel(const std::array<double, Dimension> &v){
            for(size_t i=0; i < Dimension; ++i) vel[i] = v[i];
        }

        /**
         * @brief Setter method for property of the particle
         * @param p Property of particle: mass for gravitational force, charge for Coulomb force
        */
        void setProperty(double p){
            property = p;
        }

        /**
         * @brief Getter method for property of the particle
         * @return property Property of particle: mass for gravitational force, charge for Coulomb force
        */
        double getProperty() const {
            return property;
        }

        const std::array<double, Dimension>& getAccel() const {
            return accel;
        }

        /**
         * @brief Getter method for positions of the particle
         * @return pos Array of positions of the particle
        */
        const std::array<double, Dimension>& getPos() const{
            return pos;
        }

        /**
         * @brief Getter method for velocity of the particle
         * @return vel Array of velocity of the particle
        */
        const std::array<double, Dimension>& getVel() const{
            return vel; 
        }

        /**
         * @brief Getter method for force of the particle
         * @return force Array of force of the particle
        */
        const std::array<double, Dimension>& getForce() const{
        return force;
        }


        /**
         * @brief Getter method for the ID of the particle
         * @return id ID of the particle
        */
        int getId() const{
            return id;
        }

        /**
         * @brief Getter method for the radius of the particle
         * @return radius Radius of the particle
        */
        double getRadius() const{
            return radius;
        }

        /**
         * @brief Getter method for the ID of the particle
         * @return type Type of the particle: 0 for gravitational, 1 for Coulomb 
        */
        bool getType() const{
            return type;
        }

        //method that resets total force for next implementation
        void resetForce() {
            for (size_t i = 0; i < Dimension; ++i ) {
                force[i] = 0.0; 
            }
        }

        //method that calculates the square of the distance
        double squareDistance(const Particle<Dimension> &p) const{
            double square_dist = 0.0;
            for(size_t i = 0; i < Dimension; ++i) {
                const double difference = pos[i] - p.getPos()[i];
                square_dist += difference * difference;
            }
            
            return square_dist;
        }

        /**
         * @brief Method that adds force
         * @param force_qk Array of components of the force between particles q and k
         * 
        */
        void addForce(const std::array<double, Dimension> &force_qk) {

            for(size_t i = 0; i < Dimension; ++i){
                force[i] += force_qk[i];
            }

        }

        /**
         * @brief Method that updates positions and velocities using Euler integration
         * @param delta_t Time step after which the state of the particle is being updated
         * 
        */
        void update(const double delta_t) {
            for(size_t i = 0; i < Dimension; ++i) 
            {
                pos[i] += vel[i] * delta_t;
                vel[i] += (force[i] / ((property<0)? -property:property)) * delta_t;
            }
        }

        void velocityVerletUpdate(const double delta_t) {
            std::array<double, Dimension> prevAccel;
            for(size_t i=0; i<Dimension; ++i)
            {
                prevAccel[i] = accel[i];
                accel[i] = (force[i] / ((property<0)? -property:property));
                pos[i] += vel[i] * delta_t + 0.5 * prevAccel[i] * delta_t * delta_t;
                vel[i] += 0.5 * (prevAccel[i] + accel[i]) * delta_t;
            } 
        }

        /**
         * @brief Method that updates positions and velocities using Euler integration and resets force
         * @param delta_t Time step after which the state of the particle is being updated
         */
        void updateAndReset(const double delta_t) {

            for(size_t i = 0; i < Dimension; ++i) pos[i] += vel[i] * delta_t;
            for(size_t i = 0; i < Dimension; ++i) vel[i] += (force[i] / property) * delta_t;

            resetForce();
        }

        /**
         * @brief Method that prints info about the particles
         */
        void printStates() const{
            std::cout << "Id: " << id << std::endl;
            
            std::cout << "Position: ";
            for (size_t i = 0; i < Dimension; ++i) {
                std::cout << pos[i];
                if (i < Dimension - 1) {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;

            (!type? std::cout << "Mass: " : std::cout << "Charge: ");
            std::cout << property << std::endl;

            std::cout << "Force: ";
            for (size_t i = 0; i < Dimension; ++i) {
                std::cout << force[i];
                if (i < Dimension - 1) {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;
            
            std::cout << "Velocity: ";
            for (size_t i = 0; i < Dimension; ++i) {
                std::cout << vel[i];
                if (i < Dimension - 1) {
                    std::cout << " ";
                }
            }
            std::cout << std::endl;
        }

        /**
         * @brief Manages collisions between this particle and the simulation borders.
         * @param boundary Half-size of the simulation area
         */
        void manageCollision(double boundary){
            for(size_t i = 0; i < Dimension; ++i){
                if (pos[i] + radius > boundary){
                    vel[i] = -vel[i];
                    if(pos[i] > boundary)
                        pos[i] = boundary - radius;
                }
                else if (pos[i] - radius < -boundary){
                    vel[i] = -vel[i];
                    if(pos[i] < -boundary)
                        pos[i] = -boundary + radius;
                }
            }
        }

        /**
         * @brief Manages an elastic collision with another particle.
         * @param p Other particle involved in the collision
         * @param softening Factor used to determine whether the particles overlap
         */
        void manageCollision(Particle<Dimension> &p, double softening = 1.0){
            if (&p != this && squareDistance(p) <
                (radius + p.getRadius()) * (radius + p.getRadius()) * softening){
                const double distanceSquared = squareDistance(p);
                const double firstMass = std::abs(property);
                const double secondMass = std::abs(p.getProperty());
                //caso in cui le particelle si sovrappongono completamente
                if (distanceSquared == 0.0 || firstMass == 0.0 || secondMass == 0.0) {
                    return;
                }
                //calcolo della distanza tra le particelle e del vettore normale alla collisione
                const double distance = std::sqrt(distanceSquared);
                std::array<double, Dimension> normal{};
                for (size_t i = 0; i < Dimension; ++i) {
                    normal[i] = (p.getPos()[i] - pos[i]) / distance;
                }
                //calcolo della velocità relativa lungo la normale alla collisione
                double relativeNormalVelocity = 0.0;
                for (size_t i = 0; i < Dimension; ++i) {
                    relativeNormalVelocity += (p.getVel()[i] - vel[i]) * normal[i];
                }
                //controllo se le particelle si stanno allontanando o avvicinando
                if (relativeNormalVelocity >= 0.0) {
                    return;
                }
                //calcolodell'impulso della collisione
                const double impulse = -2.0 * relativeNormalVelocity /
                                       (1.0 / firstMass + 1.0 / secondMass);
                std::array<double, Dimension> newVelocity = vel;//calcolo della nuova velocità delle particelle dopo la collisione
                std::array<double, Dimension> otherNewVelocity = p.getVel();//nuova velocità dell'altra particella
                for (size_t i = 0; i < Dimension; ++i) {
                    newVelocity[i] -= impulse * normal[i] / firstMass;//aggiornamento della velocità della particella corrente
                    otherNewVelocity[i] += impulse * normal[i] / secondMass;//aggiornamento della velocità dell'altra particella
                }

                setVel(newVelocity);
                p.setVel(otherNewVelocity);
            }
        }

        /**
         * @brief Default destructor of Particle class
        */
        ~Particle(){}
    private:
        int id;
        double property;
        std::array<double, Dimension> pos;
        std::array<double, Dimension> vel;
        std::array<double, Dimension> force;
        std::array<double, Dimension> accel;
        bool type;
        double radius;
};

#endif