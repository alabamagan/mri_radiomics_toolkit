# BB-RENT

## Explanation for the Repeated Elastic Net Technique (RENT)

Based on the idea of ensembles, the authors focused the distribution of features weights (i.e., coefficients) of elastic net regularized models. The criteria for features selection in RENT were based on:

1. How often is a features selected? ($\tau_1$)
2. To which degree do the feature weights alternate between positive and negative values? ($\tau_2$)
3. Are feature weights significantly unequal to 0? ($\tau_3$)

## Original RENT[^1]

```mermaid
flowchart LR
  A[(Training<br>data)] 
  subgraph RENT["Repeated Elastic Net Technique for Feature Selection (RENT)"]
  	BS[Bootstrap<br>sampling<br>K-times]
  	T1[(Subset 1)]
  	T2[(Subset 2)]
  	Tdot((...))
	T3[(Subset K)]

  	EN1(Model 1)
  	EN2(Model 2)
  	EN3(Model N)
  	BS --> T1 & T2 & T3
    BS --- Tdot
  	T1 --> EN1
  	T2 --> EN2
  	T3 --> EN3  

    Criteria[Evaluate<br>selection<br>Criteria]
    EN1 & EN2 & EN3 --> Criteria
	style Tdot fill:none,stroke:none
  	linkStyle 3 stroke:none  
  end
  A ---> BS
  Criteria --> S[Selected features<br>subset]

```

### Weakness

The selection power of this method wasn't enough, after the initial filtering, the features left all have strong t-test performance between the two class.  The three selection criteria wasn't enough as the coefficients of the remaining features all showed high $\tau_1$ and $\tau_3$. Although $\tau_2$ can be used to filter away around 60% of features, different features were kept each time RENT is performed on the bootstrapped dataset (i.e. different data from the same set is giving different feature set).

Also, it is not rare that the some of the repeated elastic net models trained couldn't converge, but the features from these unconverged runs were still included by RENT. While this is related to the non-optimal alpha and L1-ratio, this is still problematic.

> **Notes:** Later it is found that while `RENT_Classification` showed unstableness, `RENT_Regression` seems to show better selection stability. However, when the outer K-fold is repeated again, the selected features were not always the same. Furthermore, after the features were recalculated from the original data together with some added features, the selected features changes, so it is still sensitive to the data some how but more stable than doing nothing. Therefore, we proposed to boost the elastic net prior to evaluating the three selection criteria.

### Usage

```python
from RENT import RENT
# Define setting for RENT
model = RENT.RENT_Classification(data=train_data, 
                                 target=train_labels, 
                                 feat_names=train_data.columns, 
                                 C=[10], # 1/alpha; needs to be a list
                                 l1_ratios=[0.5], # needs to be a list
                                 autoEnetParSel=False, # if not set to false the C/L1 list is permuted
                                 poly='OFF',
                                 testsize_range=(0.25,0.25),
                                 scoring='mcc',
                                 classifier='logreg',
                                 K=100,
                                 random_state=0,
                                 verbose=1)
model.train()
selected_features = model.select_features(tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975)
```



## Modifying and implementing BB-RENT

First the original RENT is modified such that instead of running the elastic net on each bootstrapped subset once, the boosted elastic net is used. 

```mermaid
flowchart LR
  A[(Training<br>data)] 
  subgraph RENT["Boosted-RENT"]
  	BS[Bootstrap<br>sampling<br>K-times]
  	T1[(Subset 1)]
  	T2[(Subset 2)]
  	Tdot((...))
	T3[(Subset K)]

  	EN1(Weak 1.1)
  	EN2(Weak 2.1)
  	EN3(Weak N.1)

  	EN12(Weak 1.2)
  	EN22(Weak 2.2)
  	EN32(Weak N.2)
  
    EN1n(Weak 1.K)
  	EN2n(Weak 2.K)
  	ENn(Weak N.K)
  	BS --> T1 & T2 & T3
    BS --- Tdot
  	T1 --> EN1 --> EN12 --> EN1n
  	T2 --> EN2 --> EN22 --> EN2n
  	T3 --> EN3 --> EN32 --> ENn

    Criteria[Evaluate<br>selection<br>Criteria]
    EN1n & EN2n & ENn --> Criteria
	style Tdot fill:none,stroke:none
  	linkStyle 3 stroke:none  
  end
  A --> BS
  Criteria --> S[Selected features<br>subset]
```

Then, the Boosted-RENT is used multiple times on the bootstrapped subset of the training data, which then becomes the bagged-boosted RENT (BB-RENT)

```mermaid
flowchart LR
	A[(Training<br>data)] 
	subgraph BB-RENT
		Bootstrapper --> |Repeat N times|IF(Initial feature filtering) --> BRENT(Boosted-RENT)
		BRENT --> SF[/Selected Features/] --> Bootstrapper
	end
	A --> BB-RENT
	BB-RENT --> E[/"Features with selection<br> frequency > 0.5N"/]
```

### Usage

TODO