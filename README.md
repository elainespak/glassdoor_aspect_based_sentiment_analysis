# ABSA with Glassdoor company reviews
Company review websites, such as [Glassdoor](https://www.glassdoor.com/index.htm), are a platform for employees to anonymously write about their work experience.
In other words, they enable job hunters to *shop* for a company based on others' reviews, just like customers shopping for items on online shopping malls.

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

I collected Glassdoor review data of every company that belonged to S&P 500 at least once from year 2008 to 2018.

Some of the text mining techniques that I have applied so far:
- Word2Vec from [this paper](https://arxiv.org/pdf/1301.3781.pdf)
- Latent aspect rating analysis from [this paper](https://www.cs.virginia.edu/~hw5x/paper/rp166f-wang.pdf)
