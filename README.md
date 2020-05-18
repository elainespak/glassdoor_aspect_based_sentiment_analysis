# ABSA with Glassdoor company reviews
Company review websites, such as Glassdoor, allow job hunters to *shop* for a company based on anonymous reviews, just like customers shopping for items from online shopping malls.

I collected Glassdoor review data of every company who belonged to S&P 500 at least once from year 2008 to 2018.

**Glassdoor data is extremely rich, both in quantity and dimensions.** A user can anonymously write about her experience in [pros], [cons], and [advice] categories, and rate her experience in the following 8 aspects:
- overall
- culture and value
- career opportunities
- senior leadership
- work-life balance
- compensation and benefits
- approval of CEO
- recommendation to friend
- business outlook.

Therefore, it is possible to take apart a written text and analyze it in the above aspects.

Some of the text mining techniques that I have applied so far:
- Word2Vec from [this paper](https://arxiv.org/pdf/1301.3781.pdf)
- Latent aspect rating analysis from [this paper](https://www.cs.virginia.edu/~hw5x/paper/rp166f-wang.pdf)
