SELECT
    crsp_a_stock.stkdelists.delactiontype,
    crsp_a_stock.stkdelists.permno,
    crsp_a_stock.stkdelists.delistingdt,
    id_table.primaryexch,
    id_table.nasdissuno,
    id_table.siccd,
    crsp_a_stock.stkdelists.deldtprc,
    crsp_a_stock.stkdelists.deldtprcflg,
    crsp_a_stock.stkdelists.delstatustype,
    crsp_a_stock.stkdelists.delreasontype,
    crsp_a_stock.stkdelists.delpaymenttype,
    crsp_a_stock.stkdelists.delpermno,
    crsp_a_stock.stkdelists.delpermco,
    crsp_a_stock.stkdelists.delret,
    crsp_a_stock.stkdelists.delretmisstype,
    crsp_a_stock.stkdelists.delnextdt,
    crsp_a_stock.stkdelists.delnextprc,
    crsp_a_stock.stkdelists.delnextprcflg,
    crsp_a_stock.stkdelists.delamtdt,
    crsp_a_stock.stkdelists.deldivamt,
    crsp_a_stock.stkdelists.deldistype,
    crsp_a_stock.stkdelists.deldlydt

FROM crsp_a_stock.stkdelists
LEFT JOIN (
    SELECT
        permno,
        primaryexch,
        nasdissuno,
        siccd
    FROM crsp_a_stock.stksecurityinfohdr
) AS id_table 

ON crsp_a_stock.stkdelists.permno = id_table.permno

WHERE crsp_a_stock.stkdelists.delistingdt BETWEEN
    '2014-01-01'::date AND '2025-09-30'::date

AND ("crsp_a_stock"."stkdelists"."delactiontype" = ANY (ARRAY['GDR','GEX','GLI','LOS','MER']))
