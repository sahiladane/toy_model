import random as r
import matplotlib.pyplot as plt

prior_alice = 0.5
prior_bob = 0.5

fox_truthful_alice = 0.2
fox_truthful_bob = 0.8

# Experiment


left_given_truthful_right = 0.4

left_given_lies_right = 0.5

left_given_truthful_left = 0

left_given_lies_left = 0.75

# assume that theta^*=(R,T)

def bayes_updating_l(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob):

        # posterior after observing left 

        posterior_alice_l = prior_alice * ( (1 - fox_truthful_alice) * left_given_lies_right + fox_truthful_alice * left_given_truthful_right)/ ( prior_alice * ( (1 - fox_truthful_alice) * left_given_lies_right + fox_truthful_alice * left_given_truthful_right) + (1-prior_alice) * ( (1 - fox_truthful_alice) * left_given_lies_left + fox_truthful_alice * left_given_truthful_left) )

        #print(posterior_alice_l)

        posterior_bob_l = prior_bob * ( (1 - fox_truthful_bob) * left_given_lies_right + fox_truthful_bob * left_given_truthful_right)/ ( prior_bob * ( (1 - fox_truthful_bob) * left_given_lies_right + fox_truthful_bob * left_given_truthful_right) + (1-prior_bob) * ( (1 - fox_truthful_bob) * left_given_lies_left + fox_truthful_bob * left_given_truthful_left) )

        #print(posterior_bob_l)

        fox_posterior_alice_l = fox_truthful_alice * ( prior_alice * left_given_truthful_right + (1-prior_alice) * left_given_truthful_left) / (fox_truthful_alice * ( prior_alice * left_given_truthful_right + (1-prior_alice) * left_given_truthful_left) + (1-fox_truthful_alice) * ( prior_alice * left_given_lies_right + (1-prior_alice) * left_given_lies_left) )

        #print(fox_posterior_alice_l)

        fox_posterior_bob_l = fox_truthful_bob * ( prior_bob * left_given_truthful_right + (1-prior_bob) * left_given_truthful_left) / (fox_truthful_bob * ( prior_bob * left_given_truthful_right + (1-prior_bob) * left_given_truthful_left) + (1-fox_truthful_bob) * ( prior_bob * left_given_lies_right + (1-prior_bob) * left_given_lies_left) )

        #print(fox_posterior_bob_l,'\n')

        return posterior_alice_l, posterior_bob_l, fox_posterior_alice_l, fox_posterior_bob_l

def bayes_updating_r(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob):

        # posteriors after observing right

        posterior_alice_r = prior_alice * ((1 - fox_truthful_alice) * (1-left_given_lies_right) + fox_truthful_alice * (1-left_given_truthful_right))/ ( prior_alice * ( (1 - fox_truthful_alice) * (1-left_given_lies_right) + fox_truthful_alice * (1-left_given_truthful_right)) + (1-prior_alice) * ( (1 - fox_truthful_alice) * (1-left_given_lies_left) + fox_truthful_alice * (1-left_given_truthful_left)) )

        #print(posterior_alice_r)

        posterior_bob_r = prior_bob * ( (1 - fox_truthful_bob) * (1-left_given_lies_right) + fox_truthful_bob * (1-left_given_truthful_right))/ ( prior_bob * ( (1 - fox_truthful_bob) * (1-left_given_lies_right) + fox_truthful_bob * (1-left_given_truthful_right)) + (1-prior_bob) * ( (1 - fox_truthful_bob) * (1-left_given_lies_left) + fox_truthful_bob * (1-left_given_truthful_left)) )

        #print(posterior_bob_r)

        fox_posterior_alice_r = fox_truthful_alice * ( prior_alice * (1-left_given_truthful_right) + (1-prior_alice) * (1-left_given_truthful_left)) / (fox_truthful_alice * ( prior_alice * (1-left_given_truthful_right) + (1-prior_alice) * (1-left_given_truthful_left)) + (1-fox_truthful_alice) * ( prior_alice * (1-left_given_lies_right) + (1-prior_alice) * (1-left_given_lies_left)) )

        #print(fox_posterior_alice_r)

        fox_posterior_bob_r = fox_truthful_bob * ( prior_bob * (1-left_given_truthful_right) + (1-prior_bob) * (1-left_given_truthful_left)) / (fox_truthful_bob * ( prior_bob * (1-left_given_truthful_right) + (1-prior_bob) * (1-left_given_truthful_left)) + (1-fox_truthful_bob) * ( prior_bob * (1-left_given_lies_right) + (1-prior_bob) * (1-left_given_lies_left)) )

        #print(fox_posterior_bob_r,'\n')

        return posterior_alice_r, posterior_bob_r, fox_posterior_alice_r, fox_posterior_bob_r

alice_event_posteriors = []
alice_fox_posteriors = []
bob_event_posteriors = []
bob_fox_posteriors = []
signal_realizations =[]
axis_points = []

iterations = 1000
for i in range(iterations):
        # p= probability of observing r
        ex = r.binomialvariate(n=1,p=(1-left_given_truthful_right))
        signal_realizations.append(ex)
        axis_points.append(i)
        if ex == 1:
                a, b, c, d = bayes_updating_r(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob)
                prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob = a, b, c, d
                alice_event_posteriors.append(prior_alice)
                alice_fox_posteriors.append(fox_truthful_alice)
                bob_event_posteriors.append(prior_bob)
                bob_fox_posteriors.append(fox_truthful_bob)
        else:
                a, b, c, d = bayes_updating_l(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob)
                prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob = a, b, c, d
                alice_event_posteriors.append(prior_alice)
                alice_fox_posteriors.append(fox_truthful_alice)
                bob_event_posteriors.append(prior_bob)
                bob_fox_posteriors.append(fox_truthful_bob)

print(prior_alice)
print(fox_truthful_alice)
print(prior_bob)
print(fox_truthful_bob, "\n")

print(sum(alice_event_posteriors) / len(alice_event_posteriors))

axis_points = [i/iterations for i in axis_points]

print(alice_fox_posteriors[-1])

#print(signal_realizations)
plt.scatter(axis_points, alice_event_posteriors, c = "blue")
#plt.scatter(axis_points, alice_fox_posteriors, c= "orange")
plt.scatter(axis_points, bob_event_posteriors, c = "red")
#plt.scatter(axis_points, bob_fox_posteriors, c= "yellow")


plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Time in [0,1]")
plt.ylabel("Posterior")
plt.show()

