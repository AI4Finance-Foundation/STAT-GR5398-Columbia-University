"""Explicit second-machine review decisions for INTC baseline; no model/API calls.

These records were authored after reading every immutable source unit and its
original evidence, under docs/machine_adjudication_protocol.md. They are judgments,
not a program that infers grounding from token overlap. Raw judge labels and
verifier decisions are not inputs to these decisions.
"""
import json
from pathlib import Path
from grounding_audit import load_inputs, source_units

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = {}


def atom(claim, classification, lines=(), reason="", numeric="not_numeric", kind="factual"):
    return {"claim":claim,"claim_type":kind,"classification":classification,"numeric_status":numeric,
            "evidence":[{"line_start":a,"line_end":b} for a,b in lines],"reason":reason}


def U(*claims, reason="The writer prompt contains financial/peer tables and one earnings-news excerpt; it supplies no evidence for this specific business, causal or risk proposition.", numeric="not_numeric", kind="factual"):
    return [atom(c,"Unsupported",reason=reason,numeric=numeric,kind=kind) for c in claims]


def S(claim, lines, reason, numeric="not_numeric", kind="factual"):
    return [atom(claim,"Supported",lines,reason,numeric,kind)]


def C(claim, lines, reason, numeric="not_numeric", kind="factual"):
    return [atom(claim,"Contradicted",lines,reason,numeric,kind)]


def put(section, unit, *groups, nonfactual=""):
    DECISIONS.setdefault(section,{})[unit]={"unit_id":unit,"atoms":[a for group in groups for a in group],"nonfactual_reason":nonfactual}


F=((4,4),(6,6)); CM=((4,4),(10,10)); EM=((4,4),(12,12)); EPS=((4,4),(15,15)); PE=((4,4),(16,16)); SG=((4,4),(13,13))
Q=((42,47),); PEV=((30,36),); PEB=((19,27),)

put('tagline',0,S('Intel has positive contribution margins.',CM,'All actual contribution margins are positive; robust is evaluative.'),S('Intel has material positive EBITDA.',((4,4),(11,11)),'Actual EBITDA is positive and measured in billions.'),S('Intel annual revenue recently declined.',F,'2021A through 2025A revenue falls each year.'))
put('tagline',1,S('Intel SG&A efficiency has recently improved as measured by SG&A/revenue.',SG,'Ratio declines from 11.1% in 2022A to 8.7% in 2025A; future declines remain assumptions.'),U('Intel practices disciplined cost management.','SG&A efficiency supports Intel operational resilience.',kind='causal'))
put('tagline',2,U('Intel has strong fundamentals.','Intel faces market fluctuations.',reason='The favorable fundamentals and market-state assertions are not established by these data; attractive entry is an investment opinion.'))

put('company_overview',0,U('Intel is a global technology leader.','Intel is a multinational company.','Intel specializes in semiconductor manufacturing.','Intel specializes in computing innovation.'))
put('company_overview',1,U('Intel designs microprocessors.','Intel produces microprocessors.','Intel designs and produces integrated circuits.','Intel designs and produces related hardware components.'))
put('company_overview',2,U('Intel has a significant global semiconductor industry position.','Intel serves original equipment manufacturers.','Intel serves cloud service providers.','Intel serves enterprise customers.','Intel serves customers in North America.','Intel serves customers in Europe.','Intel serves customers in Asia.','Intel customer base is diversified.'))
put('company_overview',3,U('Intel has a broad product portfolio.','Intel has substantial market presence.'),S('Intel generates tens of billions in annual revenue.',F,'All actual annual revenue values are between 52.853 and 79.024 billion.'),S('Intel annual revenue recently declined.',F,'Actual revenue declines across 2021A–2025A.'),U('Industry cyclicality caused Intel revenue declines.','Competitive pressure caused Intel revenue declines.',kind='causal'))
put('company_overview',4,U('Intel designs semiconductor products.','Intel manufactures semiconductor products.','Intel sells semiconductor products.','Intel sells microprocessors for personal computers.','Intel sells microprocessors for servers.','Intel sells microprocessors for data centers.'))
put('company_overview',5,U('Intel receives product sales revenue from Client Computing.','Intel receives product sales revenue from Data Center.','Intel receives product sales revenue from Internet of Things.','Intel receives product sales revenue from Mobileye.','Mobileye provides autonomous-driving technology.'))
put('company_overview',6,U('Intel strategy emphasizes innovation leadership.','Intel strategy expands manufacturing capacity.','Intel strategy diversifies into emerging technology sectors.'))
put('company_overview',7,U('Intel aims to regain technological leadership.','Intel invests in advanced process nodes.','Intel invests in new architectures.','Intel expands foundry services for other chip designers.'))
put('company_overview',8,S('Intel 2023 revenue was approximately 54.2 billion.',F,'2023A is 54.228 billion, rounded to one decimal billion.','correct','quantitative'),U('Client Computing was Intel largest revenue segment in 2023.','Data Center ranked after Client Computing in 2023 revenue.',numeric='unverifiable'))
put('company_overview',9,S('Intel revenue declined compared with prior years.',F,'Annual actual revenue declines.'),U('Intel invests heavily in new product development.','Intel invests heavily in capacity expansion.','Intel invests in next-generation processors.','Intel invests in chip packaging technologies.'))
put('company_overview',10,U('Intel focuses on artificial intelligence markets.','Intel focuses on autonomous-vehicle markets.','Intel focuses on 5G infrastructure markets.','Artificial intelligence, autonomous vehicles and 5G are high-growth markets.',numeric='unverifiable'))
put('company_overview',11,U('Intel was founded in 1968.',numeric='unverifiable'),U('Intel pioneered microprocessor development.','Intel expanded into data-center technology.','Intel made recent acquisitions to strengthen artificial intelligence capabilities.','Intel made recent acquisitions to strengthen autonomous-driving capabilities.'))
put('company_overview',12,U('Intel current CEO is Pat Gelsinger.','Pat Gelsinger has a deep semiconductor-technology background.','Pat Gelsinger previously served as VMware CEO.'),U('Pat Gelsinger returned to lead Intel in early 2021.',numeric='unverifiable'))
put('company_overview',13,U('Under Pat Gelsinger Intel refocused on manufacturing excellence.','Under Pat Gelsinger Intel refocused on technological innovation.',kind='causal'))
put('company_overview',14,U('Intel CFO has extensive finance and industry experience.','Other Intel executives have extensive finance and industry experience.','Executive experience supports Intel strategic transformation.',kind='causal'))
put('company_overview',15,S('Intel 2023 revenue was 54.2 billion.',F,'2023A revenue 54.228 billion rounds to 54.2.','correct','quantitative'),S('Intel 2023 EPS was low relative to its earlier visible actual years.',EPS,'EPS 0.4 in 2023A is below 4.89/1.95 in 2021A/2022A.'),S('Intel profitability margins fluctuate.',EM,'Actual EBITDA margins vary materially.'),U('Industry challenges significantly affected Intel net income.',kind='causal'))
put('company_overview',16,U('Intel market capitalization is substantial.','Intel has a foundational semiconductor-sector role.','Its foundational role explains its market capitalization.',kind='causal'))
put('company_overview',17,U('Intel stock price has experienced volatility.','Intel trading range reflects investor concerns about competitive dynamics.','Intel trading range reflects investor concerns about industry cycles.',kind='causal'),U('Intel has an observed 52-week trading range characterizing these concerns.',numeric='unverifiable'))
put('company_overview',18,U('Intel operates in semiconductors.','AMD competes with Intel.','NVIDIA competes with Intel.','AMD has gained high-performance-computing market share.','NVIDIA has gained graphics-processing market share.'))
put('company_overview',19,U('Intel ranks among the leading global semiconductor companies.','Intel faces intense competition.','Intel faces market-share pressure.'))
put('company_overview',20,U('The semiconductor industry innovates rapidly.','Semiconductor demand is cyclical.','Semiconductors require significant capital spending.','These industry characteristics shape Intel strategic priorities.','These industry characteristics shape Intel financial performance.',kind='causal'))

put('investment_overview',0,nonfactual='Investment-thesis status heading.')
put('investment_overview',1,nonfactual='Section heading.')
put('investment_overview',2,S('Intel reported Q2 2026 revenue of 16.1 billion.',Q,'News title and line45 establish quarter and revenue.','correct','quantitative'),S('Intel reported Q2 2026 revenue growth of 25% year over year.',Q,'News line45 directly reports 25%.','correct','quantitative'))
put('investment_overview',3,S('Intel reported Q2 2026 GAAP EPS of -2.16.',Q,'News line46 loss attributable to Intel is $(2.16).','correct','quantitative'),S('Intel reported Q2 2026 non-GAAP EPS of 0.42.',Q,'News line46 explicitly labels non-GAAP.','correct','quantitative'),S('Intel reported a GAAP loss in the quarter.',Q,'Negative GAAP per-share earnings establish reported unprofitability.'))
put('investment_overview',4,S('Intel guided Q3 2026 revenue to 15.8–16.8 billion.',Q,'News line47 gives the forecast range.','correct','quantitative'),S('Intel guided Q3 2026 non-GAAP EPS to 0.38.',Q,'Explicit line47 non-GAAP forecast.','correct','quantitative'),U('Intel guidance indicates management confidence in continued improvement.',kind='causal'))
put('investment_overview',5,S('Intel latest quarter revenue grew.',Q,'News line45 year-over-year growth.'),S('Intel historical revenue CAGR is -9.6%.',F,'For 2021A–2025A, (52.853/79.024)^(1/4)-1=-9.57%, displayed -9.6%.','correct','quantitative'),S('Intel historical EPS weakened.',EPS,'4.89 falls to -0.0589, with 2024A -4.38.'),S('The pipeline projects negative Intel EPS through 2028.',EPS,'Forecast EPS -0.06/-0.07/-0.07; substantial is subjective.','correct','quantitative'))
put('investment_overview',6,S('Intel EBITDA margin fell to 2.3% in 2024.',EM,'2024A displays2.3%.','correct','quantitative'),S('Intel EBITDA margin is forecast around 30% by 2028.',EM,'2028E30.6% supports an approximate toward30% statement.','correct','quantitative'))
put('investment_overview',7,nonfactual='Section heading.')
put('investment_overview',8,S('Intel reported Q2 2026 revenue of16.1billion.',Q,'Line45 within Q2news.','correct','quantitative'),S('Intel reported Q2 2026 revenue grew25% year over year.',Q,'Line45.','correct','quantitative'),S('That positive quarterly growth contrasts with recent annual revenue declines.',((4,6),(42,45)),'Actual annual revenues2021–2025 decline; news quarterly growth positive.'))
put('investment_overview',9,S('Intel reported Q2 GAAP EPS -2.16.',Q,'Line46 reported loss.','correct','quantitative'),S('Intel remains unprofitable on reported quarterly GAAP earnings.',Q,'Negative GAAP EPS establishes a loss.'))
put('investment_overview',10,S('Intel reported non-GAAP EPS0.42.',Q,'Line46 explicitly reports value.','correct','quantitative'),U('Positive non-GAAP EPS establishes improving underlying operations.',reason='No previous-quarter/non-GAAP comparison or operating bridge appears in the prompt.',kind='causal'))
put('investment_overview',11,C('Q3 revenue guidance anticipates only stable to slightly higher revenue.',Q,'Against Q2 16.1, range15.8–16.8 also permits a1.86% decline; omits downside of range.'),S('Q3 guidance anticipates positive non-GAAP EPS.',Q,'Line47 forecast0.38 is positive.'),U('Guidance reflects management cautious optimism.',kind='causal'))
put('investment_overview',12,S('Intel revenue fell substantially between2021 and2025.',F,'79.024 to52.853 billion.','correct','quantitative'),S('The pipeline forecasts a modest subsequent revenue recovery.',((4,6),(14,14)),'2026E–2028E growth5/6/4% remains below2021A level.'))
put('investment_overview',13,S('Intel actual EBITDA margins compressed to a2.3% low in2024.',EM,'42.9/33.8/20.7/2.3/27.2 actuals;2.3 lowest.','correct','quantitative'),S('Pipeline EBITDA margin is expected to improve after the low.',EM,'2025A rebounds27.2 and forecasts27.6/29.1/30.6.'))
put('investment_overview',14,U('Intel profitability remains challenged relative to AMD and NVIDIA.',reason='Peer table gives EBITDA amounts, not peer net profitability/margins or comparable EPS.'),S('Intel latest comparable EV/EBITDA is below AMD and NVIDIA.',PEV,'2025 Intel14.4982 vsAMD47.8456 andNVDA33.7774; latest shared actual year.'),S('Intel latest reported quarter revenue grew.',Q,'News line45.'))
put('investment_overview',16,U('Intel preexisting core investment thesis relies on operational-efficiency improvement.','Intel preexisting thesis relies on scale and R&D to regain competitiveness.','Intel key markets include data centers.','Intel key markets include PC processors.',reason='The original evidence has no prior investment thesis, R&D, business-segment or competitive-position record.'),S('Intel annual revenue declines need stabilization.',F,'Actual annual revenue decreases; need is evaluative.'))
put('investment_overview',15,nonfactual='Section heading.')
put('investment_overview',17,S('Intel latest quarter revenue growth is positive.',Q,'25% YoY.'),S('Intel latest quarter non-GAAP earnings are positive.',Q,'Non-GAAP EPS0.42.'),U('These results show Intel turnaround initiatives gaining traction.',reason='No named initiatives, execution evidence or attribution bridge is provided.',kind='causal'))
put('investment_overview',18,S('Intel has recent reported GAAP losses.',((4,4),(15,15),(42,46)),'Annual2024/2025 EPS negative and quarterly loss in news.'),C('The negative EPS outlook through2028 is historical.',EPS,'2026E–2028E are explicit forecasts, not historical observations.','incorrect','quantitative'))
put('investment_overview',19,S('Intel historical revenue CAGR is negative.',F,'Displayed -9.6% actual endpointCAGR.'),S('Intel historical EBITDA margins were depressed.',EM,'2024A2.3%, below earlier years.'),U('These statistics establish persistent structural challenges.',kind='causal'))
put('investment_overview',20,U('Intel guidance has improved.',reason='Only one guidance vintage is supplied, with no prior forecast for comparison.'),U('Intel Q2 results beat expectations.',reason='No analyst/management prior expectation appears in the writer evidence.'),U('Intel faces execution risk.','Intel faces adverse competitive dynamics.'))
put('investment_overview',21,nonfactual='Normative conditions for confirming an investment thesis; no additional empirical assertion.')
put('investment_overview',22,U('Intel faces ongoing headwinds.',reason='Broad external headwinds not specified by the limited evidence; thesis review status is the writer opinion.'))

put('valuation_overview',0,U('Intel valuation is shaped by significant operational challenges.','Intel valuation is shaped by evolving market dynamics.',kind='causal'))
put('valuation_overview',1,S('Intel historical PE fluctuated widely.',PE,'Actual ratios10.49,13.55,125.625,-4.63,-614.60.'),S('Intel2021 PE was near10.',PE,'10.49284 is near10.','correct','quantitative'),S('Intel historical PE peaked above120 in2023.',PE,'125.625 maximum among shown actuals.','correct','quantitative'),S('Intel PE became negative after2023 as EPS became negative.',((4,4),(15,16)),'2024A/2025A PE and EPS negative; ratio sign agrees with negative earnings.','correct','quantitative'))
put('valuation_overview',2,S('Intel earnings weakened while revenue and EBITDA margins fell from earlier actual levels.',((4,6),(12,12),(15,16)),'2021A to2025A earnings/revenue/margin lower; association only.'),U('Severe earnings pressure caused the full observed valuation volatility.',reason='Ratio arithmetic sign is visible, but attribution of magnitude/volatility lacks contemporaneous prices and decomposition.',kind='causal'))
put('valuation_overview',3,S('Intel2021 revenue was approximately79billion.',F,'79.024billion rounds79.','correct','quantitative'),S('Intel2025 revenue was approximately53billion.',F,'52.853billion rounds53.','correct','quantitative'),C('The2025 revenue value is an estimate.',F,'Table explicitly labels2025A actual.','incorrect','quantitative'),S('Intel2021–2025 revenue compounded down nearly10% annually.',F,'(52.853/79.024)^(1/4)-1=-9.57%.','correct','quantitative'))
put('valuation_overview',4,S('Intel actual EPS became negative in2024 and remained negative in2025.',EPS,'2024A -4.38 and2025A -0.0589.','correct','quantitative'),S('Negative EPS indicates reported earnings deterioration relative to earlier positive years.',EPS,'Earlier actual EPS4.89/1.95/0.4 positive.'))
put('valuation_overview',5,S('Intel PE is negative in its latest actual observations.',PE,'2024A/2025A negative.'),S('Intel EV/EBITDA spiked in2024 and fell in2025.',PEV,'22.4823 in2023,106.938 in2024,14.4982 in2025.','correct','quantitative'),U('The identified financial stresses materially caused the full change in valuation metrics.',reason='No enterprise-value or price decomposition proves causation.',kind='causal'))
put('valuation_overview',6,U('Intel valuation is relatively depressed against AMD and NVIDIA on a forward basis.',reason='Peer EV/EBITDA table contains historical years only, and no peer forward PE/target valuations.'))
put('valuation_overview',7,C('NVIDIA EV/EBITDA multiples remain consistently higher than Intel.',PEV,'2024 NVDA43.4522 is lower thanINTC106.938; the unqualified persistence claim has a counterexample.'),U('NVIDIA higher multiples reflect stronger growth prospects.','NVIDIA higher multiples reflect better profitability metrics.',kind='causal'))
put('valuation_overview',8,S('AMD EV/EBITDA multiples are generally higher than Intel.',PEV,'AMD higher in4 of5 shared years;2024 exception prevents always but permits generally.'),U('AMD higher multiples reflect market recognition of its improving competitive position.',kind='causal'))
put('valuation_overview',9,S('Intel latest actual earnings are low to negative.',EPS,'Actual EPS2024/2025 negative.'),S('Intel actual margins are lower than earlier actual years.',EM,'2025A27.2% below2021A42.9%.'),U('Intel earnings and margins explain its valuation discount to peers.',reason='A valuation causal decomposition is absent.',kind='causal'))
put('valuation_overview',10,U('Intel valuation indicates market caution about near-term recovery.','Intel valuation indicates market concern about strategic execution.',kind='causal'))
put('valuation_overview',11,U('Competitive pressure drives Intel valuation.','Technological transitions drive Intel valuation.','Execution problems eroded Intel market share.','Execution problems eroded Intel profitability.',kind='causal'))
put('valuation_overview',12,U('Semiconductors undergo rapid innovation.','Advanced process technologies face intense competition.','Chip design faces intense competition.'))
put('valuation_overview',13,U('Intel historically dominated its industry.','Intel dominance is being challenged.'),S('Intel revenue and margin profile deteriorated from earlier actual years.',((4,6),(12,12)),'Revenue and EBITDA margin2025 below2021.'),U('Financial deterioration proves challenges to historical dominance.',kind='causal'))
put('valuation_overview',14,U('Operational headwinds dampened market sentiment.','Cyclical downturns reduced semiconductor demand.',kind='causal'))
put('valuation_overview',15,U('Intel valuation reflects market expectations of a protracted recovery.','Intel valuation lacks a bubble or scarcity premium.',reason='No market expectations, fair-value benchmark or premium decomposition is present.'))
put('valuation_overview',16,U('Depressed multiples signal market caution about Intel turnaround sustainability.','Depressed multiples signal market caution about margin forecasts.',kind='causal'))
put('valuation_overview',17,U('Intel execution risks can impair profitability.','Structural semiconductor-industry shifts can impair Intel profitability.',kind='causal'))
put('valuation_overview',18,U('Intel faces fundamental challenges in its competitive environment.',reason='Reasonable valuation is opinion; the competitive premise is not evidenced by peer selection alone.'))
put('valuation_overview',19,U('Intel depressed multiples indicate tempered market expectations for near-term earnings.','Intel depressed multiples indicate tempered market expectations for growth.',kind='causal'))
put('valuation_overview',20,U('Intel valuation appropriately incorporates operational execution risks.','Intel valuation appropriately incorporates industry competition risks.',kind='causal'))
put('valuation_overview',21,U('Intel current low valuation provides a margin of safety.','Intel market evolves rapidly.',reason='No intrinsic-value benchmark, share price or market-change evidence supports these premises; advice to weigh uncertainty is opinion.'))

put('risks',0,U('Intel faces material risks to future revenue.','Intel faces material risks to future profitability.','Intel faces material risks to future valuation.',reason='Introductory company-specific exposure assertions are not established merely by historical figures.'))
put('risks',1,nonfactual='Organization of the author risk categories, not an independently testable company fact.')
put('risks',2,S('Intel historical annual revenue declined.',F,'Actual revenue falls2021A–2025A.'),C('Intel revenue CAGR from2021 to2028 is-9.6%.',F,'2028E61.17840456/2021A79.024 over7years implies about-3.59%; -9.6% belongs2021A–2025A. Includes forecast endpoint.','incorrect','quantitative'),U('Decreasing customer demand explains Intel revenue decline.','Macroeconomic headwinds explain Intel revenue decline.',kind='causal'))
put('risks',3,U('Semiconductors are transforming rapidly.','Emerging technologies threaten Intel traditional business model.','Shifting end-market requirements threaten Intel traditional business model.',kind='causal'))
put('risks',4,U('Global economic cycles cause Intel demand volatility.','Computing technology shifts increase Intel market risk.','Data-center technology shifts increase Intel market risk.',kind='causal'))
put('risks',5,S('AMD and NVIDIA have stronger multiyear EBITDA growth than Intel.',PEB,'AMD2021→2025 rises4.166→7.275bn whileIntel falls33.874→14.354; NVDA2022→2025 rises11.351→86.137 whileIntel falls21.299→14.354. Recent is interpreted multiyear, not every annual interval.'),U('Higher peer EV/EBITDA represents superior competitive quality.',reason='Multiple magnitude does not establish superiority or competitive risk.'),U('Intel competitive risks are acute relative to those peers.',reason='EBITDA/multiple tables do not establish product competition or risk intensity.'))
put('risks',6,U('More innovative competitor products threaten Intel market share.','More cost-effective competitor products threaten Intel market share.',kind='causal'))
put('risks',7,U('Intel faces material technological-obsolescence exposure.','Semiconductor manufacturing advances rapidly.','Chip design advances rapidly.'))
put('risks',8,U('Failure to keep pace erodes Intel competitive position.','Failure to keep pace erodes Intel pricing power.','Those failures could cause damaging price wars.',kind='causal'))
put('risks',9,U('Intel supply chain depends on rare materials.','Intel faces fabrication-capacity constraints.','Geopolitical disruption affects Intel global suppliers.','These supply vulnerabilities are common across semiconductor manufacturing.'))
put('risks',10,U('Intel historically struggled to transition process nodes.','Intel historically struggled to transition manufacturing technologies.','Those transition problems create significant execution risk.',kind='causal'))
put('risks',11,U('Loss of specialized personnel threatens Intel innovation.','Loss of specialized personnel threatens Intel operational efficiency.',kind='causal'))
put('risks',12,S('Intel profitability is under pressure compared with earlier actual years.',((4,4),(12,12),(15,15)),'2025 EBITDA margin and EPS below2021, with negative recent EPS; prominent risk is evaluative.'))
put('risks',13,S('Intel EBITDA margin was42.9% in2021.',EM,'2021A42.9%.','correct','quantitative'),C('Intel2024 EBITDA margin2.3% is a forecast.',EM,'The numeric value matches but2024A is explicitly actual.','incorrect','quantitative'),S('Intel margin recovery is anticipated in forecast years.',EM,'2026E27.6 to2028E30.6 forecast improvement.'))
put('risks',14,S('Intel recent actual EPS turned negative.',EPS,'2024A/2025A negative.'),U('Negative EPS establishes problems creating shareholder value.',reason='No shareholder total return, cost-of-capital or economic-profit evidence is supplied.',kind='causal'))
put('risks',15,S('Intel earnings are volatile.',EPS,'Actual EPS4.89,1.95,0.4,-4.38,-0.0589 swings materially.'),U('Intel relies on capital markets to finance strategic investments.','Intel faces debt-covenant risk.','Intel faces cash-flow-adequacy risk if markets tighten.',reason='No funding mix, covenants, cash-flow data or capital-market access evidence appears in writer prompt.'))
put('risks',16,U('New export regulations may affect Intel.','New data-privacy regulations may affect Intel.','New environmental regulations may affect Intel.'))
put('risks',17,U('Intel faces litigation risk.','Intel faces reputational-damage risk.','Intel operates in politically sensitive jurisdictions.'))
put('risks',18,U('Intel semiconductor manufacturing creates environmental liabilities.','Intel energy use adds regulatory complexity.',kind='causal'))
put('risks',19,S('Intel historical revenue declined steeply.',F,'79.024→52.853bn actual decline.'),S('Intel historical profitability is volatile.',((4,4),(12,12),(15,15)),'Margins and EPS vary substantially.'),U('These patterns threaten valuation.','These patterns threaten investor confidence.',kind='causal'))
put('risks',20,U('Intel has a core competitive-position risk relative to AMD.','Intel has a core competitive-position risk relative to NVIDIA.','Failure to innovate would accelerate Intel market-share losses.',kind='causal'))
put('risks',21,U('Intel manufacturing-transition execution threatens future growth.','Intel manufacturing-transition execution threatens cost management.',kind='causal'))
put('risks',22,S('Intel has recent negative earnings.',EPS,'Actual2024/2025 EPS negative.'),S('Intel margins have compressed versus earlier years.',EM,'Actual2025 below2021.'),U('Intel depends on capital markets.','These financial vulnerabilities constrain Intel strategic flexibility.',kind='causal'))
put('risks',23,U('Intel has a strong brand.','Intel has significant resources.',reason='No brand evidence or resource/capital data in the writer prompt; risk-management recommendations are normative.'))
put('risks',24,U('An evolving regulatory landscape threatens disruption to Intel.',reason='No regulation/impact evidence accompanies this normative recommendation.'))
put('risks',25,U('Intel has market leadership that could be lost.','Unaddressed challenges could prolong Intel underperformance.',kind='causal'))

put('competitor_analysis',0,U('Intel operates in semiconductor manufacturing.','The semiconductor industry is highly competitive.','Semiconductors innovate rapidly.','Semiconductors require significant capital investment.','A few dominant players have intense rivalry.'))
put('competitor_analysis',1,U('AMD is a primary Intel competitor.','NVIDIA is a primary Intel competitor.','TSMC is an emerging competitor.','TSMC is primarily a foundry.','TSMC manufacturing prowess threatens Intel.'))
put('competitor_analysis',2,U('AMD competes prominently with Intel in CPUs.'))
put('competitor_analysis',3,U('AMD gained market share.','Ryzen processors caused AMD market-share gains.','EPYC processors caused AMD market-share gains.','Ryzen and EPYC received favorable performance reception.','Ryzen and EPYC received favorable efficiency reception.',kind='causal'))
put('competitor_analysis',4,U('AMD advanced chip architecture confers a strategic advantage.','AMD has a strong product portfolio.','AMD challenges Intel consumer-segment dominance.','AMD challenges Intel enterprise-segment dominance.'))
put('competitor_analysis',5,U('AMD partners with major OEMs.','High-performance computing demand is growing.','AMD benefited from growing high-performance computing demand.',kind='causal'))
put('competitor_analysis',6,U('NVIDIA is primarily known for GPUs.','NVIDIA competition with Intel is increasing.','Intel is expanding into discrete GPUs.','Intel is expanding into AI accelerators.'))
put('competitor_analysis',7,U('NVIDIA dominates GPUs.','NVIDIA leads AI technologies.','NVIDIA leads deep-learning technologies.','NVIDIA has a robust integrated software/hardware ecosystem.'))
put('competitor_analysis',8,U('NVIDIA innovates rapidly.','NVIDIA expands into data centers.','NVIDIA expands into autonomous vehicles.','NVIDIA expansion pressures Intel expansion plans.',kind='causal'))
put('competitor_analysis',9,U('TSMC does not directly compete with Intel on branding.','TSMC does not directly compete with Intel on product sales.','TSMC advanced manufacturing poses an emerging threat to Intel.'))
put('competitor_analysis',10,U('TSMC manufactures chips for AMD.','TSMC manufactures chips for NVIDIA.','TSMC continually advances process-technology boundaries.'))
put('competitor_analysis',11,U('Intel has manufacturing challenges.','Intel has delayed process-node transitions.','Intel delays enabled TSMC to gain an edge.','TSMC edge increases foundry competition for Intel.',kind='causal'))
put('competitor_analysis',12,U('Intel expands its foundry services.','Intel foundry expansion aims to counter TSMC.','Intel foundry expansion faces steep competition.'))
put('competitor_analysis',13,U('Intel historically operated an integrated-device-manufacturing moat.','Intel historically led CPU design.','Intel has a vast customer and supplier ecosystem.'))
put('competitor_analysis',14,U('Intel benefits from brand recognition.','Intel benefits from proprietary technology.','Intel has a broad patent portfolio.'))
put('competitor_analysis',15,U('Intel makes significant R&D investments.','Intel makes significant fabrication-facility investments.','Those investments create competitor entry barriers.',kind='causal'))
put('competitor_analysis',16,U('Intel competitive moat is under pressure.'))
put('competitor_analysis',17,S('Intel2021 revenue was about79billion.',F,'2021A79.024bn.','correct','quantitative'),S('Intel2025 revenue was about53billion.',F,'2025A52.853bn.','correct','quantitative'),C('The2025 revenue value is an estimate.',F,'2025A explicitly denotes actual.','incorrect','quantitative'),U('Intel revenue decline reflects lost market share.','Intel revenue decline reflects competitive challenges.',kind='causal'))
put('competitor_analysis',18,S('Intel contribution margin contracted from earlier actual levels.',CM,'2021A55.4 versus2025A34.8.'),S('Intel EBITDA margin contracted from earlier actual levels.',EM,'2021A42.9 versus2025A27.2.'))
put('competitor_analysis',19,S('Intel EPS became negative in recent actual years.',EPS,'2024A/2025A negative.'),S('Negative EPS shows reported profitability challenges.',EPS,'Negative EPS directly establishes net loss per share.'))
put('competitor_analysis',20,S('AMD EBITDA improved on a multiyear basis.',PEB,'2021 4.166bn to2025 7.275bn; operational metrics is narrowly interpreted as supplied EBITDA.'),S('NVIDIA EBITDA improved on a multiyear basis.',PEB,'2022 11.351bn to2026 144.552bn.'),U('AMD market position improved.','NVIDIA market position improved.',reason='EBITDA growth is not evidence of market share or competitive ranking.'))
put('competitor_analysis',21,U('Intel moat is narrowing.','Manufacturing delays eroded Intel cost leadership.','Manufacturing delays eroded Intel technology leadership.','Competitor innovations gained ground.','Computing demand is shifting to specialized solutions.','Computing demand is shifting to heterogeneous solutions.','Intel is less dominant in those solutions.',kind='causal'))
put('competitor_analysis',22,S('Intel has large revenue scale.',F,'Annual revenue exceeds50bn.'),U('Intel invests in next-generation process technology.','Intel has foundry expansion initiatives.','Intel has product diversification initiatives.','Those initiatives provide a stable base to strengthen its moat.',kind='causal'))
put('competitor_analysis',23,U('AMD poses fierce competition to Intel.','NVIDIA poses fierce competition to Intel.','TSMC poses a strategic manufacturing threat to Intel.'))
put('competitor_analysis',24,U('Intel has substantial competitive advantages.','Execution challenges weaken Intel advantages.','Aggressive competitor advances weaken Intel advantages.',kind='causal'))
put('competitor_analysis',25,U('Intel moat is narrowing.','Intel manufacturing capabilities have suffered lost confidence.',reason='Conditional advice about restoring a moat embeds these unsupported present-state premises.'))

put('major_takeaways',0,S('Intel2021 revenue was79.0billion.',F,'79.024 rounds79.0.','correct','quantitative'),S('Intel2024 revenue was53.1billion.',F,'53.101 rounds53.1.','correct','quantitative'),C('Intel2021–2024 revenue CAGR was-9.6%.',F,'(53.101/79.024)^(1/3)-1≈-12.41%; -9.6% belongs2021–2025.','incorrect','quantitative'))
put('major_takeaways',1,U('Challenging market conditions drove Intel revenue contraction.','Competitive pressure drove Intel revenue contraction.','Product-demand shifts drove Intel revenue contraction.',kind='causal'),S('Intel annual revenue shrank for several years.',F,'Actual decline2021–2025.'))
put('major_takeaways',2,S('The pipeline forecasts revenue growth4–6% starting2026.',((4,4),(14,14)),'2026E5%,2027E6%,2028E4%; forecast explicitly stated.','correct','quantitative'))
put('major_takeaways',3,U('Intel revenue decline undermines its market position.','Peer revenue has grown.',reason='The prompt shows peer EBITDA, not peer revenue or market shares.'),U('Revenue decline establishes structural competitiveness problems.','Revenue decline establishes poor market-share retention.',kind='causal'))
put('major_takeaways',4,S('Intel contribution margin was55.4% in2021.',CM,'2021A55.4%.','correct','quantitative'),S('Intel contribution margin fell to32.7% in2024.',CM,'2024A32.7%.','correct','quantitative'))
put('major_takeaways',5,U('Pricing pressure contributed to margin erosion.','Investment increases contributed to margin erosion.','Inefficiencies contributed to margin erosion.',kind='causal'),S('Intel cost of operations rose between2021 and2024.',((4,4),(7,7)),'35.209bn in2021 to35.756bn in2024; not a monotonic annual rise.'))
put('major_takeaways',6,U('Margin contraction establishes weaker pricing power.','Margin contraction establishes operational challenges.','Those challenges erode Intel moat.',kind='causal'))
put('major_takeaways',7,S('Intel contribution margin improved in2025 after2024.',CM,'32.7 to34.8 actual.','correct','quantitative'),C('Contribution margin37.8% by2028 is an observed improvement.',CM,'37.8 belongs2028E forecast, not observed actual.','incorrect','quantitative'),S('Projected37.8% is below earlier contribution-margin levels.',CM,'2021–2023actual55.4/42.6/40.0 exceed37.8forecast.','correct','quantitative'),U('This margin path proves diminished business leverage.',kind='causal'))
put('major_takeaways',8,S('SG&A margin was8.3% in2021.',SG,'2021A8.3.','correct','quantitative'),S('SG&A margin reached a visible peak11.1% in2022.',SG,'Largest shown actualSGA percentage.','correct','quantitative'),C('SG&A margin has already declined to7.2% by2028.',SG,'7.2% is2028E; past-tense trajectory mixes future assumptions and observations.','incorrect','quantitative'))
put('major_takeaways',9,U('Management cost controls caused better operating efficiency.','Management cost controls caused greater scalability.',kind='causal'))
put('major_takeaways',10,S('Intel actual SG&A margin declined after2022.',SG,'11.1 to10.4/10.4/8.7 actual; flat2023/24 allowed.'),U('The decline proves better operating leverage.','Disciplined expense management caused the decline.',kind='causal'),C('Intel actual revenue has stabilized after2022.',F,'2022–2025 actual revenue declines63.054→54.228→53.101→52.853; stabilization is only projected, not established actual.'))
put('major_takeaways',11,U('Intel management focuses on expense reduction.','Expense reduction demonstrates positive management execution.',kind='causal'),S('Intel faces an actual revenue headwind.',F,'Annual actual revenue declines.'))
put('major_takeaways',12,S('Intel EBITDA margins are volatile.',EM,'Observed42.9 to2.3 to27.2 swings.'),S('Intel EBITDA margin was42.9% in2021.',EM,'2021A42.9%.','correct','quantitative'),S('Intel EBITDA margin reached a2.3% low in2024.',EM,'2024A2.3, lowest displayed actual.','correct','quantitative'),C('Intel EBITDA margin has recovered to30.6% by2028.',EM,'30.6% belongs2028E forecast, not a completed recovery.','incorrect','quantitative'))
put('major_takeaways',13,U('Intel margin contraction is more severe than AMD margin contraction.','Intel margin contraction is more severe than NVIDIA margin contraction.',reason='No peer revenue or margins provided; absolute EBITDA cannot establish margin comparisons.'),U('Relative margin deterioration establishes weaker competitive positioning.',kind='causal'))
put('major_takeaways',14,S('Intel actual margins compressed materially.',EM,'Latest2025 below2021, with2024trough.'),S('Intel EBITDA is volatile.',((4,4),(11,11)),'33.874bn to1.203bn to14.354bn actual; concern about future sustainability is opinion.'))
put('major_takeaways',15,C('Intel EV/EBITDA is elevated relative to peers throughout recent years.',PEV,'Only2024 of2022–2025 has Intel above both;2025Intel14.4982 is belowboth.'),U('Elevated EV/EBITDA demonstrates market skepticism.',kind='causal'))
put('major_takeaways',16,nonfactual='Investment-monitoring recommendation and subjective transitional-phase interpretation; no new factual premise.')

put('news_summary',0,S('The supplied latest Intel release concerns Q2 2026.',Q,'Only one supplied article, title establishesQ2; most recent is limited to supplied news inventory.','correct','quantitative'),S('Intel reported Q2 revenue growth25% year over year.',Q,'Line45.','correct','quantitative'),S('Intel reported Q2 revenue16.1billion.',Q,'Line45.','correct','quantitative'))
put('news_summary',1,S('Intel reported GAAP loss per share2.16.',Q,'Line46 negative2.16 asloss.','correct','quantitative'),S('Intel reported positive non-GAAP EPS0.42.',Q,'Line46.','correct','quantitative'),U('The GAAP/non-GAAP difference establishes ongoing operational challenges.',reason='No GAAP-to-adjusted bridge or operating mechanism is provided.',kind='causal'))
put('news_summary',2,S('Intel Q3 2026 revenue guidance is15.8–16.8billion.',Q,'Line47 range explicitly forecast.','correct','quantitative'),S('Intel Q3 2026 non-GAAP EPS guidance is0.38.',Q,'Line47.','correct','quantitative'),U('Guidance signals cautious optimism.',kind='causal'))
put('news_summary',3,S('Intel reported a Q2 2026 revenue rebound.',Q,'Quarterly25%YoY growth.','correct','quantitative'),S('Intel annual revenue declined over multiple years.',F,'Actual2021–2025 declines.'),S('Intel reported negative GAAP earnings despite positive non-GAAP results.',Q,'Line46 provides opposite EPSsigns.'),U('Guidance establishes stabilization of Intel financial performance.',reason='One forecast range, including lower quarterly revenue, does not establish realized stabilization.'))
put('news_summary',4,S('Intel EBITDA has been volatile.',((4,4),(11,11)),'Large actual swings.'),S('Intel EBITDA margin was notably low in2024.',EM,'2.3% actual trough.','correct','quantitative'),S('The pipeline projects EBITDA margin improvements through2028.',EM,'2026E27.6/2027E29.1/2028E30.6.','correct','quantitative'))
put('news_summary',5,S('NVIDIA latest comparable actual EBITDA is higher than Intel.',PEB,'2025NVDA86.137bn vsINTC14.354bn; no universal all-yearsclaim.'),S('NVIDIA latest comparable EV/EBITDA is higher than Intel.',PEV,'2025NVDA33.7774 vsINTC14.4982;2024exception prevents universalclaim.'),U('These peer figures establish ongoing competitive pressure on Intel.','These figures establish a challenging competitive landscape.',kind='causal'))
put('news_summary',6,U('Market sentiment toward Intel is cautiously optimistic.','Revenue growth caused optimistic market sentiment.','Positive non-GAAP earnings caused optimistic market sentiment.','GAAP losses temper market sentiment.','Strategic execution concerns temper market sentiment.',kind='causal'),S('Intel reported positive revenue growth and non-GAAP earnings.',Q,'Line45/46 reported quarter.'),S('Intel has recent GAAP losses.',((4,4),(15,15),(42,46)),'Latest annual and quarter negative.'),U('Intel needs strategic execution to restore growth/profitability.',reason='Financial shortfall is evidenced but the strategic mechanism is not.'))
put('news_summary',7,U('Mixed market sentiment may cause moderate near-term Intel stock volatility.','Investors are weighing operational improvements against competitive headwinds.',reason='No sentiment, stock-price history, investor survey or response evidence is supplied.',kind='causal'))


def main():
    run=ROOT/'runs/INTC_baseline_1'
    manifest, inputs=load_inputs(run)
    spec={"protocol_version":"1","run_id":manifest['run_id'],"evidence_run_id":manifest['run_id'],
          "raw_judge_audit":"audits/grounding/INTC_before_gpt41","sections":{}}
    for section,item in inputs.items():
        expected={u['unit_id'] for u in source_units(item['text'])}
        if set(DECISIONS.get(section,{}))!=expected:
            raise ValueError(f'Explicit decisions do not cover exact source units: {section}')
        spec['sections'][section]={"source_sha256":item['metadata']['source_sha256'],
            "evidence_sha256":item['metadata']['evidence_prompt_sha256'],
            "units":[DECISIONS[section][i] for i in sorted(expected)]}
    target=ROOT/'audits/machine_adjudication/INTC_before_review.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Explicit reviewed units:',sum(len(s['units']) for s in spec['sections'].values()),
          'atoms:',sum(len(u['atoms']) for s in spec['sections'].values() for u in s['units']))


if __name__=='__main__':main()
