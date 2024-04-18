prior_alice = 0.5
prior_bob = 0.5

fox_truthful_alice = 0.2
fox_truthful_bob = 0.8

# Experiment

left_given_truthful_left = 0.7
left_given_truthful_right = 0.3
left_given_lies_left = 0.3
left_given_lies_right = 0.7


def bayes_updating_r(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob):

        # posteriors after observing right

        posterior_alice_r = prior_alice * ((1 - fox_truthful_alice) * (1-left_given_lies_right) + fox_truthful_alice * (1-left_given_truthful_right))/ ( prior_alice * ( (1 - fox_truthful_alice) * (1-left_given_lies_right) + fox_truthful_alice * (1-left_given_truthful_right)) + (1-prior_alice) * ( (1 - fox_truthful_alice) * (1-left_given_lies_left) + fox_truthful_alice * (1-left_given_truthful_left)) )

        print(posterior_alice_r)

        posterior_bob_r = prior_bob * ( (1 - fox_truthful_bob) * (1-left_given_lies_right) + fox_truthful_bob * (1-left_given_truthful_right))/ ( prior_bob * ( (1 - fox_truthful_bob) * (1-left_given_lies_right) + fox_truthful_bob * (1-left_given_truthful_right)) + (1-prior_bob) * ( (1 - fox_truthful_bob) * (1-left_given_lies_left) + fox_truthful_bob * (1-left_given_truthful_left)) )

        print(posterior_bob_r)

        fox_posterior_alice_r = fox_truthful_alice * ( prior_alice * (1-left_given_truthful_right) + (1-prior_alice) * (1-left_given_truthful_left)) / (fox_truthful_alice * ( prior_alice * (1-left_given_truthful_right) + (1-prior_alice) * (1-left_given_truthful_left)) + (1-fox_truthful_alice) * ( prior_alice * (1-left_given_lies_right) + (1-prior_alice) * (1-left_given_lies_left)) )

        print(fox_posterior_alice_r)

        fox_posterior_bob_r = fox_truthful_bob * ( prior_bob * (1-left_given_truthful_right) + (1-prior_bob) * (1-left_given_truthful_left)) / (fox_truthful_bob * ( prior_bob * (1-left_given_truthful_right) + (1-prior_bob) * (1-left_given_truthful_left)) + (1-fox_truthful_bob) * ( prior_bob * (1-left_given_lies_right) + (1-prior_bob) * (1-left_given_lies_left)) )

        print(fox_posterior_bob_r,'\n')

        return posterior_alice_r, posterior_bob_r, fox_posterior_alice_r, fox_posterior_bob_r


bayes_updating_r(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob)



for i in range(10):
    a, b, c, d = bayes_updating_r(prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob)
    prior_alice, prior_bob, fox_truthful_alice, fox_truthful_bob = a, b, c, d

