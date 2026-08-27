# Qualitative readout examples

Selected by rule from the frozen panel, not by eye. Tokens marked
`*` appear in the prompt up to and including the readout position.

## genuine gain

- **model** qwen3.5-27b · **set** association · **item** rivals · **z** 0.0
- **prompt** `They shook hands for the cameras and held it a beat too long, each waiting for the other to let go first[[>>.<<]]`
- **readout token** `«.»` at index 23

| lens | top-10 readout | contextual | echo | non-echo |
|---|---|---|---|---|
| logit | `« »` `«,»` `« (»` `«-»` `«.»` `«...»` `«\n»` `«/»` `«:»` `«?»` | 1.0 | 0.0 | 1.5 |
| released-J | `«–and»` `«–»` `«�»` `«**–»` `« tossed»` `«.hm»` `«izon»` `«�»` `«籲»` `«lsa»` | 0.0 | 0.0 | 1.0 |
| released-R | `«…..»` `« though»` `«….»` `« ….»` `« they»` `« seperate»` `« …»` `« there»` `« wanting»` `« needing»` | 3.0 | 0.0 | 3.5 |

## echo driven

- **model** gemma-3-27b-it · **set** typo · **item** typo-people · **z** 0.2
- **prompt** `<bos>The stadium was packed with nearly forty thousand po[[>>eple<<]]`
- **readout token** `«eple»` at index 10

| lens | top-10 readout | contextual | echo | non-echo |
|---|---|---|---|---|
| logit | `«ized»` `«ed»` `«an»` `«al»` `«Than»` `«-»` `«fromi»` `« Tourist»` `« Min»` `«em»` | 0.5 | 0.0 | 1.0 |
| released-J | `« recieve»` `« seperate»` `« succesfully»` `« wouldnt»` `« recieved»` `« seper»` `« couldnt»` `« havent»` `« arent»` `« doesnt»` | 0.5 | 0.0 | 2.0 |
| released-R | `«,»` `« and»` `« people»` `« in»` `«'»` `« seperate»` `« (»` `« to»` `« "»` `« '»` | 3.0 | 0.5 | 0.5 |

## all poor

- **model** qwen3.5-27b · **set** poetry · **item** couplet-crack-black · **z** 0.1
- **prompt** `A rhyming couplet:\nThe storm clouds gathered with a thunder crack,[[>>\n<<]]And turned the midday sky to total`
- **readout token** `«\n»` at index 17

| lens | top-10 readout | contextual | echo | non-echo |
|---|---|---|---|---|
| logit | `« »` `«...»` `«-»` `«.»` `« (»` `«(»` `«/»` `«\u00a0»` `«\n»` `«!»` | 0.0 | 0.0 | 2.0 |
| released-J | `«ly»` `«но»` `«ся»` `«zeitig»` `«raries»` `«imately»` `«써»` `«ا»` `«jenigen»` `«lich»` | 0.0 | 0.0 | 0.5 |
| released-R | `«beau»` `«alak»` `«ly»` `«edal»` `«加拿大»` `« gotta»` `«ISTR»` `«繫»` `«zeitig»` `« Nessa»` | 0.0 | 0.0 | 0.5 |
