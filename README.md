# Exploring Employee Experience & Discovering Company Characteristics with Glassdoor Reviews
Anonymous company rating and review websites, such as [Glassdoor](https://www.glassdoor.com/index.htm), are a platform for employees to anonymously write about their work experience. **Glassdoor data is extremely rich, both in quantity and dimensions.** A user can anonymously write about her experience in the 'summary,' 'pros,' 'cons,' and 'advice' categories, and rate her experience in the following 8 aspects:
- Overall
- Culture & Value
- Career Opportunities
- Senior Management
- Work/Life Balance
- Compensation and Benefits
- Approval of CEO
- Recommendation to Friend
- Business Outlook

## Data Collection
I created a custom webscraper and collected Glassdoor review data of every company that belonged to S & P 500 at least once from year 2008 to 2018.
Please refer to the 'webscrape' folder.


## Text Preprocessing
1. Select only English reviews
   - Pre-trained FastText model (considered a SOTA for language identification as of July, 2020) (for Windows, best way to download is to get the appropriate wheel file at [this website](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext) and run `pip install ~wheelname~.whl`)
   - Among 'summary,' 'pros,' 'cons,' and 'advice,' select the longest text. Check if the longest text is predicted as English or not. If yes, keep the review. If not, discard it.
2. Remove stopwords and punctuations
3. Make every letter lower case
4. Create bigrams and trigrams
5. Apply stemming

**Result: 1,401,126 sentences from 741 companies**


## Aspect Extraction
1. Apply Chi-Square statistics from [Latent Aspect Rating Analysis (LARA)](https://www.cs.virginia.edu/~hw5x/paper/rp166f-wang.pdf)
   - Select relevant aspect: culture and value; career opportunities; senior leadership; work-life balance; compensation and benefits; ~~approval of CEO~~; ~~recommendation to friend~~; business outlook
   - Seed words:
      - culture and value: `['cultur', 'valu']`
      - career opportunities: `['career', 'opportun']`
      - senior leadership: `['senior', 'leadership', 'manag']`
      - work-life balance: `['life_bal', 'life', 'balanc']`
      - compensation and benefits: `['compens', 'benefit']`
      - business outlook: `['busi', 'outlook', 'futur']`
   - Final words:
      - culture and value: `['cultur', 'valu', 'compani', 'core', 'corpor', 'strong', 'add', 'collabor', 'larg', 'big', 'within', 'move', 'innov', 'stabl', 'size', 'chang', 'organ', 'forward', 'global', 'promot', 'year', 'constant', 'polit', 'world', 'huge', 'structur', 'last', 'ago', 'past', 'layoff', '5', 'organiz']`
      - career opportunities: `['career', 'opportun', 'advanc', 'growth', 'path', 'learn', 'grow', 'lot', 'room', 'limit', 'develop', 'new', 'plenti', 'progress', 'train', 'technolog', 'skill', 'potenti', 'slow', 'curv', 'program', 'provid', 'latest', 'invest', 'profession', 'intern', 'softwar', 'tool', 'hire', 'process', 'cross', 'talent']`
      - senior leadership: `['senior', 'leadership', 'manag', 'upper', 'store', 'level', 'middl', 'micro', 'team', 'member', 'poor', 'district', 'lack', 'assist', 'entri', 'commun', 'execut', 'mid', 'director', 'leader', 'lead', 'account', 'incompet', 'support', 'direct', 'decis', 'lower', 'staff', 'advic', 'style', 'sr', 'listen', 'top']`
      - work-life balance: `['life_bal', 'life', 'balanc', 'work', 'famili', 'worklif', 'home', 'person', 'flexibl', 'hour', 'schedul', 'hard', 'environ', 'day', 'weekend', 'week', 'long', 'time', 'shift', 'holiday', 'fun', 'full', 'part', 'night', 'enjoy', 'vacat', 'get', 'done', 'paid', 'sick', 'easi', 'peopl', 'rid']`
      - compensation and benefits: `['compens', 'benefit', 'great', 'good', 'pay', 'health', 'packag', '401k', 'place', 'decent', 'insur', 'match', 'salari', 'low', 'competit', 'discount', 'averag', 'need', 'start', 'dental', 'medic', 'compar', 'overal', 'mani', 'increas', 'bonu', 'wage', 'make', 'perk', 'sometim', 'like', 'ok']`
      - business outlook: `['busi', 'outlook', 'futur', 'unit', 'model', 'run', 'analyst', 'uncertain', 'line', 'bottom', 'front', 'financi', 'oper', 'custom', 'strategi', 'servic', 'rude', 'focu', 'sale', 'deal', 'repres', 'rep', 'associ', 'goal', 'sell', 'product', 'specialist', 'solut', 'consult', 'focus', 'client', 'floor', 'unrealist']`
3. Apply [Attention-Based Aspect Extraction (ABAE)](https://www.aclweb.org/anthology/P17-1036.pdf)
   - Original code published and generously shared by the authors [here](https://github.com/ruidan/Unsupervised-Aspect-Extraction). Check the link for specific steps.
   - Note for Windows users: The code must run with Python 2.7; For training, run the following to avoid errors: `set THEANO_FLAGS=device=cuda,floatX=float32 & python train.py --emb ../preprocessed_data/$domain/w2v_embedding --domain $domain --aspect-size $k -o output_dir -v $vocab_size`; Similarly, for evaluation, run the following: `set THEANO_FLAGS=device=cuda,floatX=float32 & python evaluation.py --emb ../sample_data/$domain/w2v_embedding --domain $domain --aspect-size $k -v $vocab_size`
   - For our purposes, we reached the best result when `k=12` and `vocab_size=12000`
   - Final words on Pros (truncated to the first 10 words):
      - `['benefit', '401k_matching', 'insurance', 'eap', 'benfits', 'tuition_reimbursement', 'medical_dental_vision', 'benifits', 'espp', 'profit_sharing']`
      - `['employee', 'leadership', 'management', 'input', 'sincere', 'constructive', 'evident', 'transparency', 'subordinate', 'strongly']`
      - `['hour', 'schedule', 'time', 'scheduled', 'weekend', 'unpaid', 'appointment', 'day', 'home', 'weekday']`
      - `['pay', 'wage', 'salary', 'payout', 'compensation', 'rate', 'paying', 'payouts', 'commision', 'hourly']`
      - `['breakroom', 'catering', 'cooky', 'freebie', 'salon', 'merch', 'themed', 'wardrobing', 'swag', 'clothing']`
      - `['midtown', 'location', 'office', 'centrally_located', 'london', 'hq', 'conveniently_located', 'located', 'tampa', 'boston']`
      - `['quit', 'told', 'didnt', 'anyway', 'wont', 'sad', 'eventually', 'remember', 'somewhere', 'saying']`
      - `['people', 'colleague', 'coworkers', 'teammate', 'atmosphere', 'environment', 'ppl', 'collegues', 'enviroment']`
      - `['research', 'analysis', 'analytics', 'instrumentation', 'analytic', 'implementation', 'database', 'deployment', 'computing', 'instrument']`
      - `['company', 'compnay', 'conglomerate', 'financial_institution', 'corporation', 'telco', 'comapny', 'merck', 'symantec', 'firm']`
      - `['opportunity', 'possibility', 'opportunites', 'opportunties', 'opps', 'avenue', 'oportunities', 'oppurtunities', 'oppertunities', 'oppurtunity']`
      - `['ruin', 'stunning', 'stepped', 'transformed', 'ruined', 'unprofessional', 'reflection', 'appeared', 'mantra', 'heaven']`
   - Final words on Cons (truncated to the first 10 words):
      - `['reorgs', 'headcount_reduction', 'orgs', 'reorganization', 'restructurings', 'reorg', 'restructures', 'restructuring', 'rifs', 'reorganisation']`
      - `['company', 'corporation', 'qualcomm', 'firm', 'eastman', 'hilton', 'raytheon', '3m', 'institution', 'compnay']`
      - `['business', 'integrate', 'creation', 'executing', 'operational', 'architecture', 'product', 'implementation', 'delivering', 'execute']`
      - `['hour', 'weekday', 'week', 'work', 'day', 'noon', 'workday', 'shift', 'peak_season', 'monday_friday']`
      - `['really', 'think', 'honestly', 'laugh', 'love', 'gonna', 'want', 'know', 'guess', 'smile']`
      - `['management', 'managment', 'mgmt', 'mgt', 'mangement', 'managemnt', 'leadership', 'managament', 'managerment', 'managent']`
      - `['pay', 'salary', 'wage', '401k_matching', 'compensation', 'bonus', 'payout', 'profit_sharing', 'substantially', 'payouts']`
      - `['promotion', 'opportunity', 'skillset', 'advancement', 'mentoring', 'upward', 'candidate', 'position', 'role', 'lateral']`
      - `['po', 'register', 'supposed', 'submitted', 'backup', 'receipt', 'confirmation', 'verify', 'ordered', 'promo']`
      - `['stressfull', 'demanding', 'strenuous', 'exhausting', 'physically_exhausting', 'unfulfilling', 'fast_paced', 'unnecessarily', 'hectic', 'stressful']`
      - `['employee', 'passenger', 'costumer', 'customer', 'associate', 'employes', 'baristas', 'guest', 'coworkers', 'employess']`
      - `['resembles', 'shabby', 'describes', 'transformed', 'egypt', 'disgraceful', 'dot_com', 'infested', 'refers', 'ultra_conservative']`
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
