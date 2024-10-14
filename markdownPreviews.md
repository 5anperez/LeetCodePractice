`live_metal_prices`:


```python


def latest_in_chosen_currency(requestedsymbols: str, requestedcurrency: str):
    """Real-time Gold, Silver, Palladium, Platinum and 160+ currency rates based on selected Currency.

    Args:
        requestedsymbols: The requested symbols for the metal prices.
        requestedcurrency: The requested currency for the metal prices.

    Returns:
        A dictionary containing the latest metal prices in the chosen currency.
    """
    

def latest_selected_metals_in_selected_currency_in_grams(requestedunitofweight: str, requestedcurrency: str, requestedsymbols: str):
    """Real-time Gold, Silver, Palladium, Platinum and 160+ currency rates based on selected Currency.

    Args:
        requestedunitofweight: The requested unit of weight.
        requestedcurrency: The requested currency for the metal prices.
        requestedsymbols: The requested symbols for the metal prices.

    Returns:
        A dictionary containing the latest metal prices in the chosen currency and unit of weight.
    """
    

def latest_retrieve_xau_xag_pa_pl_eur_gbp_usd():
    """Real-time Gold, Silver, Palladium and Platinum prices delivered in USD, GBP and EUR.

    Returns:
        A dictionary containing the latest metal prices in USD, GBP, and EUR.
    """
    

def latest_retrieve_selected_160_symbols(requestedsymbols: str):
    """Real-time Gold, Silver, Palladium, and Platinum provided in 160+ currencies including USD, GBP and EUR.

    Args:
        requestedsymbols: The requested symbols for the metal prices.

    Returns:
        A dictionary containing the latest metal prices in the requested symbols.
    """
    

```












































`google_news`:


```python


def supported_languages_and_regions():
    """This endpoint is used to retrieve a list of supported languages and regions.
    """
    

def suggest(keyword: str, lr: str='en-US'):
    """This endpoint is used to get autocomplete suggestions or query predictions as a user types a search query. 

    The endpoint requires the **keyword** parameter, which represents the partial text entered by the user. 
    You can send a request with the partial text, and the request will generate a JSON response containing 
    a list of relevant autocomplete suggestions for the search query.

    Args:
        keyword: The mandatory parameter to specify the search term
        lr: language region, ex: en-US
    """
    

def search(keyword: str, lr: str='en-US'):
    """This endpoint is used to search for news from Google News based on keywords. 

    Args:
        keyword: The mandatory parameter to specify the search term
        lr: language region, ex: en-US
    """
    

def world(lr: str='en-US'):
    """This endpoint is used to get world news from Google News. 

    The optional parameter that can be used is "lr" to determine the region.

    Args:
        lr: language region, ex: en-US
    """
    

def sport(lr: str):
    """This endpoint is used to get sport news from Google News.

    Args:
        lr: language region, ex: en-US
    """
    

def science(lr: str):
    """This endpoint is used to get science news from Google News.

    Args:
        lr: language region, ex: en-US
    """
    

def health(lr: str):
    """This endpoint is used to get health news from Google News.

    Args:
        lr: language region, ex: en-US
    """
    

def entertainment(lr: str):
    """This endpoint is used to get entertainment news from Google News.

    Args:
        lr: language region, ex: en-US
    """
    

def business(lr: str):
    """This endpoint is used to get business news from Google News.

    Args:
        lr: language region, ex: en-US
    """
    

def latest(lr: str):
    """This endpoint is used to get the latest news from Google News.

    Args:
        lr: language region, ex: en-US
    """
    

def technology(lr: str):
    """This endpoint is used to get technology news from Google News.

    Args:
        lr: language region, ex: en-US
    """
    

```


































real_time_product_search:

```python


def search(query: str, min_rating: str=None, product_condition: str=None, max_shipping_days: int=None, store_id: str=None, on_sale: bool=None, free_returns: bool=None, free_shipping: bool=None, max_price: int=None, language: str='en', sort_by: str=None, country: str='us', min_price: int=None, page: int=None) -> dict[str, Any]:
    """Search for product offers - both free-form queries and GTIN/EAN are supported. Each page contains up to 30 product offers. Infinite pagination/scrolling is supported using the *page* parameter.

    Args:
        query: Free-form search query or a GTIN/EAN (e.g. *0194252014233*).
        min_rating: Return products with rating greater than the specified value.
            Possible values: `1`, `2`, `3`, `4`.
        product_condition: Only return products with a specific condition.
            Possible values: `NEW`, `USED`, `REFURBISHED`.
        max_shipping_days: Only return product offers that offer shipping/delivery of up to specific number of days (i.e. shipping speed).
        store_id: Only return product offers from specific stores (comma separated list of store id's). 
            Store IDs can be obtained from the Google Shopping URL after using the **Seller** filter by taking the part after the `merchagg:` variable within the `tbs` parameter.
            When filtering for a certain Seller / Store on Google Shopping, a URL similar to the following is shown on the address bar: `https://www.google.com/search?gl=us&tbm=shop&q=shoes&tbs=mr:1,merchagg:m100456214|m114373355`, 
            in that case, the Store IDs are **m100456214** and **m114373355** - to filter for these stores, set store_id=m100456214,m114373355.
        on_sale: Only return product offers that are currently on sale.
            Default: `false`.
        free_returns: Only return product offers that offer free returns.
            Default: `false`.
        free_shipping: Only return product offers that offer free shipping/delivery.
            Default: `false`.
        max_price: Only return product offers with price lower than a certain value.
        language: The language of the results.
            Valid values: see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
            Default: `en`.
        sort_by: Sort product offers by best match, top rated, lowest or highest price.
            Possible values: `BEST_MATCH`, `TOP_RATED`, `LOWEST_PRICE`, `HIGHEST_PRICE`.
            Default: `BEST_MATCH`.
        country: Country code of the region/country to return offers for.
            Valid values: see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
            Default: `us`.
        min_price: Only return product offers with price greater than a certain value.
        page: Results page to return.
            Default: `1`.

    Returns:
        A list of products (including their product_id).
    """
    

def product_offers(product_id: str, country: str='us', language: str='en') -> dict[str, Any]:
    """Get all offers available for a product.

    Args:
        product_id: Product id of the product for which to fetch offers.
        country: Country code of the region/country to return offers for.
            Valid values: see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
            Default: `us`.
        language: The language of the results.
            Valid values: see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
            Default: `en`.
    Returns:
        A list of product offers.
    """
    

def product_reviews(product_id: str, country: str='us', language: str='en', offset: str=None, rating: str=None, limit: str=None) -> dict[str, Any]:
    """Get all product reviews. Infinite pagination/scrolling is supported using the *limit* and *offset* parameters.

    Args:
        product_id: Product id of the product for which to fetch reviews.
        country: Country code of the region/country.
            Valid values: see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
            Default: `us`.
        language: The language of the results.
            Valid values: see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
            Default: `en`.
        offset: Number of reviews to skip.
            Valid values: integers from 0-30000
            Default: `0`.
        rating: Only return reviews with user rating greater than the specified value.
            Valid values: 1-5.
        limit: Maximum number of product reviews to return.
            Valid values: integers from 0-100.
            Default: `10`.
    Returns:
        A list of product reviews.
    """
    

def product_details(product_id: str, country: str='us', language: str='en') -> dict[str, Any]:
    """Get the details of a specific product by product id. Returns the full product details in addition to reviews sample, photos, product specs and more information.
    
    Args:
        product_id: Product id of the product for which to get full details.
        country: Country code of the region/country to return offers for.
            Valid values: see https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
            Default: `us`.
        language: The language of the results.
            Valid values: see https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
            Default: `en`.
    Returns:
        A list of product details.
    """
```




















amazon

```python

def search(query: str, brand: str=None, min_price: int=None, max_price: int=None, country: str='US', category_id: str='aps', sort_by: str='RELEVANCE', page: str='1') -> dict[str, Any]:
    
    """Search for product offers on Amazon.

    Args:
        query: Search query. Supports both free-form text queries or a product asin.
        brand: Find products with a specific brand. Multiple brands can be specified as a comma (,) separated list. The brand values can be seen from Amazon's search left filters panel, as seen [here](https://www.amazon.com/s?k=phone).
            **e.g.** `SAMSUNG`
            **e.g.** `Google,Apple`
        min_price: Only return product offers with price greater than a certain value. Specified in the currency of the selected country. For example, in case country=US, a value of *105.34* means *$105.34*.
        max_price: Only return product offers with price lower than a certain value. Specified in the currency of the selected country. For example, in case country=US, a value of *105.34* means *$105.34*.
        country: Sets the marketplace country, language and currency. 

            **Default:** `US`

            **Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`

        category_id: Find products in a specific category / department. Use the **Product Category List** endpoint to get a list of valid categories and their ids for the country specified in the request.

            **Default:** `aps` (All Departments)
        sort_by: Return the results in a specific sort order.

            **Default:** `RELEVANCE`

            **Allowed values:** `RELEVANCE, LOWEST_PRICE, HIGHEST_PRICE, REVIEWS, NEWEST`

        page: Results page to return.

            **Default:** `1`
    """
    

def product_details(asin: str, country: str='US') -> dict[str, Any]:
    """Get additional product information / details such as description, about, rating distribution and specs.

    Args:
        asin: Product ASIN for which to get details. Supports batching of up to 10 ASINs in a single request, separated by comma (e.g. *B08PPDJWC8,B07ZPKBL9V, B08BHXG144*).

            Note that each ASIN in a batch request is counted as a single request against the plan quota.
        country: Sets the marketplace country, language and currency. 

            **Default:** `US`

            **Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`
    """
    

def product_reviews(asin: str, query: str=None, sort_by: str=None, verified_purchases_only: bool=None, page_size: int=10, page: int=1, star_rating: str=None, images_or_videos_only: bool=None, country: str='US') -> dict[str, Any]:
    """Get and paginate through all product reviews on Amazon.

    Args:
        asin: Product asin for which to get reviews.
        query: Find reviews matching a search query.
        sort_by: Return reviews in a specific sort order.

            **Default:** `TOP_REVIEWS`

            **Allowed values:** `TOP_REVIEWS, MOST_RECENT`

        verified_purchases_only: Only return reviews by reviewers who made a verified purchase.
        page_size: Results page size.

            **Allowed values:** `1-20`

            **Default:** `10`
        page: Results page to return.

            **Default:** `1`
        star_rating: Only return reviews with a specific star rating.

            **Default:** `ALL`

            **Allowed values:** `ALL, 5_STARS, 4_STARS, 3_STARS, 2_STARS, 1_STARS, POSITIVE, CRITICAL`

        images_or_videos_only: Only return reviews containing images and / or videos.
        country: Sets the marketplace country, language and currency. 

            **Default:** `US`

            **Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`
    """
    

def product_category_list(country: str='US') -> dict[str, Any]:
    """Get Amazon product categories (per country / marketplace).

    Args:
        country: Sets the marketplace country, language and currency. 

            **Default:** `US`

            **Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`
    """
    

def product_offers(asin: str, delivery: str=None, limit: int=100, product_condition: str=None, country: str='US') -> dict[str, Any]:
    """Get top 10 offers of a specific product on Amazon by its asin.

    The first offer in the list is the pinned offer returned by the **Search** endpoint. Supports filtering by product condition.

    Args:
        asin: Product ASIN for which to get offers. Supports batching of up to 10 ASINs in a single request, separated by comma (e.g. *B08PPDJWC8,B07ZPKBL9V, B08BHXG144*).

            Note that each ASIN in a batch request is counted as a single request against the plan quota.
        delivery: [EXPERIMENTAL]
            Find products with specific delivery option, specified as a comma delimited list of the following values: `PRIME_ELIGIBLE,FREE_DELIVERY`.

            **e.g.** `FREE_DELIVERY`
            **e.g.** `PRIME_ELIGIBLE,FREE_DELIVERY`

        limit: Maximum number of offers to return.

            **Default:** `100`
        product_condition: Find products in specific conditions, specified as a comma delimited list of the following values: `NEW, USED_LIKE_NEW, USED_VERY_GOOD, USED_GOOD, USED_ACCEPTABLE`.

            **e.g.** `NEW,USED_LIKE_NEW`
            **e.g.** `USED_VERY_GOOD,USED_GOOD,USED_LIKE_NEW`

        country: Sets the marketplace country, language and currency. 

            **Default:** `US`

            **Allowed values:**  `US, AU, BR, CA, CN, FR, DE, IN, IT, MX, NL, SG, ES, TR, AE, GB, JP`
    """
    
```
