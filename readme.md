## Readme
Playing around with mlflow using a Home Insurance Kaggle Dataset. 

## Framing what I intend to do with the dataset
* Aim to predict the lapsing probability
* Lapsing directly represents cutomer retention.
* Retention directly affects estimated future margin this means the predicted probabilities are important to use alongside metrics such as estimated LTV and claims. This will help pricing calculations and it's cheaper to retain customers. 
* Lapsing probabilities can be used to directly intervene in the customer journey to improve retention. Potential uses of a lapsing probability:
    * More profitable and competitive pricing. 
    * Will aid in pro-active retention initiatives.
    * Understand and target on X-sell and Up-sell to drive longer term outcomes. 

### Technology choices
* Notebooks for rough explorations -> scripts end up in python
* Poetry for virtual environment management.
* Mlflow for tracking experiments.
* Example unit tests, there should be more.
* The code isn't production ready

## Modelling
### Benchmarking
* Dummy classifier predicting the baseline lapsed probability. 
* Often averages like this are what we've got to beat initially. 
* Provides a simple baseline to beat.

### XGBoost
* Performs well with higher feature numbers when compared to Logistic Regression.
* Great for categorical and binary predictors. 
* Probability calibration is usually good out of the box, compared to other tree based methods. 
* Good interpretability with Tree Visualisations and SHAP method

Cautions
* Overfitting
* There is a danger of 'coarse' splitting creating hard probabilities
* Covariance among features means it requires the training to work harder.
* Scaling helps make the space easier to navigate, however not done here for interpretability

#### Talking points
* Contents only insurance represents a lapsing risk (SUM_INSURED_BUILDINGS, AD_BUILDINGS)
* Adding people or products/ sub products reduces a lapsing risk (LEGAL_ADDON_POST_REN, HP2_ADDON_POST_REN)
* If a customer has already renewed they represet a lower lapsing risk
* Older customers are less likely to lapse (AGE_YEARS)
* PureDD customers are less likely to lapse 
* Older property homeowner more likely to lapse (YEARS_BUILT)

### Summary
* The model out performs the benchmark, prediction is better than using an average lapsing probability. 
<br />

* Estimate the financial implications of using the more accurate probabilities:
    * Improve existing processs: Existing processes using an average lapsing probability, may be improved by using a better calibrated figure. 
    * Save cost: Estimating savings from more accurate pricing
    * New revenue opps: Developing new intervetions to increase revenue i.e. Pro-active offers.  
<br />
* Continue to iterate until performance improves: 
    * Best focus on the data quality and features
    * Hyperparameter tuning once the data is done
    * Test/ test/ val splits need refining dependand upon deployment scenario to more accurately represent the data used and inferred in prod.  
<br />
* Once an adequate performance and business benefit can be demonstrated (less technically), share within the business. The insight gained and predications made can be used by BI teams, Marketing, Finance, Product and Partnerships. 
<br />
* Other considerations:
    * Probabilities require calibration
    * Develop clear systems architecture for consumer systems: BI/ Live Pricing/ Customer comms/ Offers/ ...
    * Live deployment, monitoring and measurement environments are needed. 
    * Move to remote repo dev > qa > prod
    * Refactor code (especially training code) for deployment within a framework: Kubeflow, for example.
    * Serving is something that could be done via API, and batch (for analysis), but again environments need provisioning for this. 

## Quickstart
### Initialise your virtual env with poetry
```console
poetry install

# To run unit tests
pytest tests
```

### Run the example experiments
```console
chmod +x run_training_experiments.sh
./run_training_experiments.sh
```

### Head over to mlflow and view the results!

```console
# open a new terminal if you still want to develop/
poetry run mlflow ui
```

