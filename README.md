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
   - Final words:
     - Aspect 0: company centene medtronic mckesson thermo metlife corp starwood kellogg uhg subsidiary corporation jacob boa boeing optum fiserv jabil inc sysco caesar prudential ecolab tyco canada anthem uhc amgen parker goodyear northern leidos merck eaton unum citi lockheed southwest conglomerate fis jnj blackrock ford traveler marsh dupont pfizer windstream franchise bb ge hca hilton labcorp smith idexx mgm raytheon expedia bank williams hartford affiliate fedex edward comcast conduent seagate centurylink ryder compnay continental citrix northrop caterpillar jpm bd covidien saic igt ctl citigroup convergys equifax whirlpool delta motorola comapny adp alcoa bofa cummins africa zimmer bny biomet indiana entity frontier stanley
     - Aspect 1: know think said tell say saying one sorry matter yeah pray wrong really kidding warn gonna something yes hey nothing exactly want anything dont screw anyways someone get thats honestly wanna never figured hear guess realise going lol youll sue wrote go see told ya screwed write right anyway somebody wouldnt cuz fired regret hope die believe remember hell nobody hate wonder didnt someday thing heck laugh otherwise actually telling knew scream oh agree probably gotta would happened complain nope youre admit mad apologize thankful literally dude simply theyre glad bother ask im read wont wake anyone bc quit forget
     - Aspect 2: employee emplyees employess employes empoyees sincere deeply subordinate dis importantly individual utmost strives strive innovation partner improving truely employe genuine truly workforce worker teamwork teammate bettering providing thoughtful authentic colleague innovative sustainability mutual sincerely patient honesty kindness foster mission staff ethical motivates associate positivity troop inclusion professional committed talent reinforce inclusiveness emphasizes empolyees humility demonstrating underling importance improve nurture ibmers creating loyalty enhancing maintaining reliability humanity delivering constructive inspires genuinely integrity philanthropic personnel openness stakeholder cultivate holistic value inclusive positively tangible excellence collaborative people consumer conscientious fostering wellbeing fairness environmentally emp demonstrate proactive intangible cultivating strong outward resourceful solving customer
     - Aspect 3: recording session recorded quiz document receipt incomplete signature password workshop submitting logging ordering counseling detailed presentation doc ect documenting verify headset ordered preparation seminar calendar po circuit fax auditing booking promptly memo tour checklist portal phone inquiry inspection library incorrect module refill submitted mail corrective arrange documentation template trial app issued log ticket ipad register validate inventory series verification logged manually refresher email emailed rx proposal electronic prompt ipads voucher deployment update cellphone tape planograms expired troubleshooting monitoring faulty sec appointment lecture supposed invoice weekly preparing sample instructor demo massage checking consultation unavailable online journal via scanning material escalation mailing
     - Aspect 4: work hour workday weekday shift weekend day overtime scheduled working schedule evening night time 5pm friday 6pm 3pm noon fri saturday week 7pm holiday overnights 11pm oncall twelve workweek summer 4pm 30pm ot flexi 10pm unpaid unscheduled 8pm 2pm 1am afternoon midnight irregular 2am 1pm 5am 9am 3am 00pm 7am 9pm 4am 6am 30am compulsory flexiable flex 10am shorten scheduling grueling 12am compressed daytime extended sunday clocked peak shorter login wok flexibile vto 8am wfh slog thursday wk mandatory exhausting manageable waking allotted flexable vacation sleep downtime clocking 00am unpredictable hectic booked wednesday accommodate graveyard clock flexibilty exhausted chore commute
     - Aspect 5: kool awesome uptight stuffy riddled friendly nice bland wonderful laidback enviornment sickening lovely drab enviorment amazing atmosphere ver cheerful cool freindly youthful chilled env fantastic thrives goo gossipy vibe enviroment suffocating juvenile lively permeates overwhelmingly ambience easygoing vibrant environement stodgy finest loving casual clicky fun benifits galore pitney ambiance beaurocratic lame vicious meh uninspiring relaxing mired dreadful stigma operative abounds liberal great eggshell blah orientated upbeat frat numbing bro existant sterile ethos stiff cliquey inherently prevails energetic tolerant clickish uplifting creepy friendliness whistle peaceful anti contagious stereotype weird sorority oriented fabulous sketchy relaxed collegiate unforgiving excellent unavoidable indecisive downer cliquish
     - Aspect 6: mgmt managament management managment oversee involvement managerial communication mgt lending support mangement consult evaluate reevaluate dictated accessibility oversight managent managemnt differs relates overseeing leadership regional indirect compensation evaluating respective standardize aligned disconnected manages disconnect reviewed manage mgr accountability divisional relation controlled disconnection operation visibility sync evaluated uniformity reporting compare administrator communicates managing penetration detached varies inline align representation communicate universal tailored delegated operational adjusted override inconsistant distribution comunication variance trickle comparison performs alignment differ dependant procurement reachable reflected echelon supervise functional administration belongs allocate empower elevated receives eg interference exposure manangement prioritized regard communicated supervision pertaining payscale attentive forecasting specifically
     - Aspect 7: organisation organization ecosystem org transformation enterprise strategy agility inertia momentum disruption mindset landscape evolution innovation acquisition complexity business portfolio footprint segment marketplace transforming behemoth nimble breakthrough centralization orgs organizational cloud firm platform maturing saas iteration pivot financials strategic methodology vertical roadmap dynamic ecommerce roadmaps functionality framework scalable margin innovate ip synergy innovator innovating execution reinventing silo profitability rapidly cognitive upstream integration telco revolution fundamental market diversifying hence model agile bu leveraging alignment sector lob diversification hurdle analytics disruptive emerge expansion startup impactful sustainable evolve company process semiconductor arena capability incremental slowing pharma globally mode operational structural conglomerate downstream technological efficiency
     - Aspect 8: etls manager managment management manger boss mgrs mangement lods egotistical unapproachable stl supervisor managemnt mgmt micromanagers asms judgmental clickish cliquish condescending atmosphere mgt coworkers personable workmate tended manangement etl cooperative approachable aloof gossipy bossy narcissistic vindictive teammate colleague tyrant environment cattiness immature unsupportive tl cliquey mate managent dictator dictatorship clicky unprofessionalism dismissive uncaring incompetent manipulative subordinate immaturity gossip staff cordial communicative irresponsible dh folk dm uneducated unprofessional jealousy backstabbing drama people arrogant catty incompetency vps sociable untrustworthy lod enviornment bos sup mgr underqualified clique ppl leadership collegues apathetic exec easygoing director insecure underling trustworthy asm spineless gm attitude empathetic eachother
     - Aspect 9: appliance jewelry furniture electronics flooring grill receiver fragrance lumber tourist neighborhood mall kitchen deli plumbing grocery cosmetic paint casino shop ringing cashiering dealer makeup shopper lighting clientele residential outdoors dairy kiosk garden packed cabinet installers mechanic photo toy bakery laundry merch overpriced cooler janitor restocking towel dusty backroom smelly salon receptionist loading costumer sweating isle decor cooking zoning stocking junk merchandise collection perfume atm installing sell unloading gadget beauty printer aisle install apparel meat cd cable cleaner restock fry stockroom packing homeowner diy trailer greasy cigarette guest vending outlet server potato homeless rack bin freezer dealership sample television restaurant softlines
     - Aspect 10: opportunity opportunites position pathway role lateral oppurtunities opportunties mentoring oppotunities oportunities advancement possibility chance career skillset laterally shadowing graduate avenue developmental mentorship placement promotion upward mentored stagnate opps qualified path candidate oppertunities grad trainee skill within advance qualification mentor training postions mobility applicant oppurtunity fresher oppertunity advancing recruited progression undergraduate networking groom education designation movement credential opp advancment selected beginner move grow transfer potential developement assignment academy educational carreer skillsets transferable programme explore promoted develop backfill pursue promotional aspiring mit experience learning growth option overqualified oportunity certification graduation supervisory domain ojt horizontally relocate specialized smes technical interviewing curriculum development recruit
     - Aspect 11: pay salary payout wage bonus measly compensation payouts commission paltry raise sti percentage rate meager comission 10k capped 5k rsus modest commision 15k 20k commisions premium yearly paid annually avg substantially paying comp deduction rsu hourly deduct income annual incentive 5x cap 3k 30k ote dividend hefty bracket earning 2k espp package cte 2k package percent maxed ctc 2x 3x threshold dollar biweekly max paycheck payed earnings accrual cent benfits spiff average lowered inflation earned increment benifits 40k bump benefit averaging averaged eligible decreased taxed 45k 60k considerably roughly vesting gainshare severance deductible 401k negotiated increase marginal eligibility pension

