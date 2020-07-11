## Dynamic Metadata Specification

Additional fields may be added, but will be ignored by Dynamic.

```
{

   "contract_url": "https://yoursite.com/more-info-about-the-coin.pdf",

   "contract_hash": "<SHA256 hash in hex of contract_url contents>",

   "contract_signature": "<Dynamic signed contract_hash>",

   "contract_address": "R9x4u22ru3zm5v8suWiXNji4ENWSG7eYkx",

   "symbol": "LEMONADE",

   "name": "Lemonade Gift",
   
   "issuer": "Lemonade Stands, Inc.",

   "description": "This coin is worth one lemonade.",

   "description_mime": "text/x-markdown; charset=UTF-8",
   
   "keywords": "Lemonade, Lemonade Stand, Gift, Cold Drink",

   "type": "Points",

   "website_url": "https://lemonadestands.com/redemption_instructions",

   "icon": "<base64 encoded png image at 32x32>",

   "image_url": "https://yoursite.com/coin-image.jpg",

   "contact_name": "Joe Schmoe",

   "contact_email": "joe_schmoe@gmail.com",

   "contact_address": "Lemonade HQ|1234 Nowhere Street|Billings, MT  83982",

   "contact_phone": "207-388-3838",

   "forsale": true,

   "forsale_price": "5000 DYN",
   
   "domain": "bitactivate.com",
   
   "restricted": "rule144"

}
```

All fields are optional. Clients, explorers, and wallets are not obligated to display or use any metadata.

### Supported Attributes

**contract_url** - Specifies the url of a document.  Might be the agreement related to the coin's purpose.

**contract_hash** - A SHA256 hash in ascii hex of the contract_url document (pdf, txt, saved html, etc).   This is so the contract, which is a link, cannot be changed without evidence of the change.  This acts as a message digest for signing.

**contract_signature** - Signed contract_hash (message digest).   Sign the contract_hash with the private key of the address that issued the asset.

**contract_address** - Address that signed contract.  Used in conjunction with the contract_signature to prove that a specific address signed the contract.  For better security, this should be the address to which the original tokens were issued.

**symbol** - The symbol.  If included, it should match the issued symbol.

**name** - The name given to the symbol.  Example: name: "Bitcoin"  Symbol: "BTC"

**description** - The description of what the symbol represents.

**description_mime** - The mime type of the description.  This may or may not be honored, depending on the client, explorer, etc.

**keywords** - helps describe an asset and allows it to be found easily Examples: "Lemonade, Cold Drink, Drinks" this will most likely be used by third party search engines. seperate keywords with commmas

**type** - The type that the qty of the token represents.  Examples: (Tokens, Points, Shares, Tickets).  This may or may not be displayed by the client.

**website_url** - The website for this token.  The client or software may or may not display this.

**icon** - Base64 encoded png at 32x32

**image_url** - Link to URL image for the coin.  Explorers or clients may not wish to display these images automatically.

**contact_name** - Name of the person or organization that owns or issued the token.

**contact_email** - The e-mail of the person or organization that owns or issued the token.

**contact_address** - The mailing address of the person or organization that owns or issued the token.  Lines should be separated by the pipe ("|") character.

**contact_phone** - The phone number of the person or organization that owns or issued the token.

**forsale** - Should be true or false.  Used by desirable token names that have been left as reissuable.  This is not for the cost of buying one token, but rather for buying the rights to own, control, and reissue the entire asset token.  This might be parsed by token broker websites.

**forsale_price** - To give buyers an idea of the cost to own and admin the asset token.   Price followed by a space, followed by the currency.  Examples: "10000 DYN" or "0.3 BTC" or "50000 USD"  This might be parsed by token broker websites.

**domain** - A root domain for the project (if applicable).  Setting the TXT record for DYN.<domain> to a signed message of the token name -- signed by the issuer address.  This could be verified by clients to ensure the token and domain go together.  Example:  Set TXT record for dyn.bitactivate.com to the signature of the message "BITACTIVATE".  Any client or individual can verify the issuer address, message "BITACTIVATE" which is the token/asset name, and valid signature in the TXT record for dyn.bitactivate.com and return true/false.

**restricted** - Designate the token as being restricted.  One example is "rule144" which means sale may be restricted because of the type of token and the exemption used for issuance.  Other restrictions types can be used here as a signal to explorers, exchanges, or token brokers.  No enforcement of restrictions is built into the Dynamic protocol. 
