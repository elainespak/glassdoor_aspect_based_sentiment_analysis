# Exploring Employee Experience & Discovering Company Characteristics with Glassdoor Reviews
Company review websites, such as [Glassdoor](https://www.glassdoor.com/index.htm), are a platform for employees to anonymously write about their work experience.
In other words, they enable job hunters to *shop* for a company based on others' reviews, just like customers shopping for items on online shopping malls.

**Glassdoor data is extremely rich, both in quantity and dimensions.** A user can anonymously write about her experience in the 'summary,' 'pros,' 'cons,' and 'advice' categories, and rate her experience in the following 8 aspects:
- overall
- culture and value
- career opportunities
- senior leadership
- work-life balance
- compensation and benefits
- approval of CEO
- recommendation to friend
- business outlook

Therefore, it is possible to take apart a written text and analyze it in the above aspects.

I collected Glassdoor review data of every company that belonged to S&P 500 at least once from year 2008 to 2018.


## Preprocessing
1. Select only English reviews
   - Pre-trained FastText model (considered a SOTA for language identification as of July, 2020) (for Windows, best way to download is to get the appropriate wheel file at [this website](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext) and run `pip install ~wheelname~.whl`)
   - Among 'summary,' 'pros,' 'cons,' and 'advice,' select the longest text. Check if the longest text is predicted as English or not. If yes, keep the review. If not, discard it.
2. Remove stopwords and punctuations
3. Make every letter lower case
4. Create bigrams and trigrams
5. Apply stemming

**Result: 1,401,126 sentences from 727 companies**


## Aspect Labeling
1. Select relevant aspect: culture and value; career opportunities; senior leadership; work-life balance; compensation and benefits; ~~approval of CEO~~; ~~recommendation to friend~~; business outlook
2. Apply [latent aspect rating analysis (LARA)](https://www.cs.virginia.edu/~hw5x/paper/rp166f-wang.pdf)
   - Original code not written in Python
   - Seed words:
      - culture and value: ['cultur', 'valu']
      - career opportunities: ['career', 'opportun']
      - senior leadership: ['senior', 'leadership', 'manag']
      - work-life balance: ['life_bal', 'life', 'balanc']
      - compensation and benefits: ['compens', 'benefit']
      - business outlook: ['busi', 'outlook', 'futur']
   - Final words:
      - culture and value: ['cultur', 'valu', 'compani', 'core', 'corpor', 'strong', 'add', 'collabor', 'larg', 'big', 'within', 'move', 'innov', 'stabl', 'size', 'chang', 'organ', 'forward', 'global', 'promot', 'year', 'constant', 'polit', 'world', 'huge', 'structur', 'last', 'ago', 'past', 'layoff', '5', 'organiz']
      - career opportunities: ['career', 'opportun', 'advanc', 'growth', 'path', 'learn', 'grow', 'lot', 'room', 'limit', 'develop', 'new', 'plenti', 'progress', 'train', 'technolog', 'skill', 'potenti', 'slow', 'curv', 'program', 'provid', 'latest', 'invest', 'profession', 'intern', 'softwar', 'tool', 'hire', 'process', 'cross', 'talent']
      - senior leadership: ['senior', 'leadership', 'manag', 'upper', 'store', 'level', 'middl', 'micro', 'team', 'member', 'poor', 'district', 'lack', 'assist', 'entri', 'commun', 'execut', 'mid', 'director', 'leader', 'lead', 'account', 'incompet', 'support', 'direct', 'decis', 'lower', 'staff', 'advic', 'style', 'sr', 'listen', 'top']
      - work-life balance: ['life_bal', 'life', 'balanc', 'work', 'famili', 'worklif', 'home', 'person', 'flexibl', 'hour', 'schedul', 'hard', 'environ', 'day', 'weekend', 'week', 'long', 'time', 'shift', 'holiday', 'fun', 'full', 'part', 'night', 'enjoy', 'vacat', 'get', 'done', 'paid', 'sick', 'easi', 'peopl', 'rid']
      - compensation and benefits: ['compens', 'benefit', 'great', 'good', 'pay', 'health', 'packag', '401k', 'place', 'decent', 'insur', 'match', 'salari', 'low', 'competit', 'discount', 'averag', 'need', 'start', 'dental', 'medic', 'compar', 'overal', 'mani', 'increas', 'bonu', 'wage', 'make', 'perk', 'sometim', 'like', 'ok']
      - business outlook: ['busi', 'outlook', 'futur', 'unit', 'model', 'run', 'analyst', 'uncertain', 'line', 'bottom', 'front', 'financi', 'oper', 'custom', 'strategi', 'servic', 'rude', 'focu', 'sale', 'deal', 'repres', 'rep', 'associ', 'goal', 'sell', 'product', 'specialist', 'solut', 'consult', 'focus', 'client', 'floor', 'unrealist']
3. Apply [attention-based aspect extraction (ABAE)](https://www.aclweb.org/anthology/P17-1036.pdf)
   ```
   @InProceedings{he-EtAl:2017:Long2,
      author    = {He, Ruidan  and  Lee, Wee Sun  and  Ng, Hwee Tou  and  Dahlmeier, Daniel},
      title     = {An Unsupervised Neural Attention Model for Aspect Extraction},
      booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
      month     = {July},
      year      = {2017},
      address   = {Vancouver, Canada},
      publisher = {Association for Computational Linguistics}
      }
   ```

   - Original code published [here](https://github.com/ruidan/Unsupervised-Aspect-Extraction) by the authors
   - On Windows, must run with Python 2.7
   - When using a novel dataset, start with `preprocess.py` and `word2vec.py`
   - Also for Windows, run `set THEANO_FLAGS=device=cuda,floatX=float32 & python train.py --emb ../preprocessed_data/$domain/w2v_embedding --domain $domain -o output_dir` to avoid errors)
   - Final words (truncated to the first 30 words):
     - Aspect 0 (Overall): company centene medtronic mckesson thermo metlife corp starwood kellogg uhg subsidiary corporation jacob boa boeing optum fiserv jabil inc sysco caesar prudential ecolab tyco canada anthem uhc amgen parker goodyear
     - Aspect 1 (?): know think said tell say saying one sorry matter yeah pray wrong really kidding warn gonna something yes hey nothing exactly want anything dont screw anyways someone get thats honestly
     - Aspect 2 (Coworkers?): employee emplyees employess employes empoyees sincere deeply subordinate dis importantly individual utmost strives strive innovation partner improving truely employe genuine truly workforce worker teamwork teammate bettering providing thoughtful authentic colleague
     - Aspect 3 (Work Resources): recording session recorded quiz document receipt incomplete signature password workshop submitting logging ordering counseling detailed presentation doc ect documenting verify headset ordered preparation seminar calendar po circuit fax auditing booking
     - Aspect 4 (Work-Life Balance): work hour workday weekday shift weekend day overtime scheduled working schedule evening night time 5pm friday 6pm 3pm noon fri saturday week 7pm holiday overnights 11pm oncall twelve workweek summer
     - Aspect 5 (Culture): kool awesome uptight stuffy riddled friendly nice bland wonderful laidback enviornment sickening lovely drab enviorment amazing atmosphere ver cheerful cool freindly youthful chilled env fantastic thrives goo gossipy vibe enviroment
     - Aspect 6 (Senior Leadership): mgmt managament management managment oversee involvement managerial communication mgt lending support mangement consult evaluate reevaluate dictated accessibility oversight managent managemnt differs relates overseeing leadership regional indirect compensation evaluating respective standardize
     - Aspect 7 (Technology?): organisation organization ecosystem org transformation enterprise strategy agility inertia momentum disruption mindset landscape evolution innovation acquisition complexity business portfolio footprint segment marketplace transforming behemoth nimble breakthrough centralization orgs organizational cloud
     - Aspect 8 (Senior Leadership): etls manager managment management manger boss mgrs mangement lods egotistical unapproachable stl supervisor managemnt mgmt micromanagers asms judgmental clickish cliquish condescending atmosphere mgt coworkers personable workmate tended manangement etl cooperative
     - Aspect 9 (Work Resources?): appliance jewelry furniture electronics flooring grill receiver fragrance lumber tourist neighborhood mall kitchen deli plumbing grocery cosmetic paint casino shop ringing cashiering dealer makeup shopper lighting clientele residential outdoors dairy
     - Aspect 10 (Career Opportunities): opportunity opportunites position pathway role lateral oppurtunities opportunties mentoring oppotunities oportunities advancement possibility chance career skillset laterally shadowing graduate avenue developmental mentorship placement promotion upward mentored stagnate opps qualified path
     - Aspect 11 (Compensation): pay salary payout wage bonus measly compensation payouts commission paltry raise sti percentage rate meager comission 10k capped 5k rsus modest commision 15k 20k commisions premium yearly paid annually avg
