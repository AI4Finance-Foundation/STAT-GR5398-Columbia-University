SELECT
    crsp_a_stock.wrds_names_query.permno,
    crsp_a_stock.wrds_names_query.permco,
    crsp_a_stock.wrds_names_query.secinfostartdt,
    crsp_a_stock.wrds_names_query.secinfoenddt,
    crsp_a_stock.wrds_names_query.securitybegdt,
    crsp_a_stock.wrds_names_query.securityenddt,
    crsp_a_stock.wrds_names_query.cusip,
    crsp_a_stock.wrds_names_query.ticker,
    crsp_a_stock.wrds_names_query.issuernm,
    crsp_a_stock.wrds_names_query.shareclass,
    crsp_a_stock.wrds_names_query.securitytype,
    crsp_a_stock.wrds_names_query.securitysubtype,
    crsp_a_stock.wrds_names_query.tradingsymbol

FROM crsp_a_stock.wrds_names_query

WHERE 1 = 1
