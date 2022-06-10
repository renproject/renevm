package clique

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum"
	"github.com/ethereum/go-ethereum/ethclient"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/params"
	"math/big"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
)

const BlockBuffer = 100
const LogDarknodeRegistered = "0x7c56cb7f63b6922d24414bf7c2b2c40c7ea1ea637c3f400efa766a85ecf2f093"
const LogDarknodeDeregistered = "0xf73268ea792d9dbf3e21a95ec9711f0b535c5f6c99f6b4f54f6766838086b842"
const LogNewEpoch = "0xaf2fc4796f2932ce294c3684deffe5098d3ef65dc2dd64efa80ef94eed88b01e"

type DNR struct {
	config         *params.CliqueConfig
	LastEpochBlock uint64                  `json:"epoch"`
	Validators     map[common.Address]bool `json:"validators"`
	synced         bool
	syncLock       sync.RWMutex
}

func NewDNR(config *params.CliqueConfig, db ethdb.Database) *DNR {
	validators := map[common.Address]bool{}
	for _, validator := range config.InitialValidators {
		validators[validator] = true
	}
	defaultDNR := &DNR{
		config:         config,
		LastEpochBlock: config.EpochBlock,
		Validators:     validators,
	}
	dnr, err := GetLatestDNR(db)
	if err == nil {
		dnr.config = config
		log.Info("loaded latest dnr from disk", "last_epoch", dnr)
		return dnr
	}
	log.Warn("failed to load latest dnr from disk, creating new drn entry with genesis data", "error", err)
	if err = defaultDNR.store(db); err != nil {
		log.Warn("failed to store default epoch in db", "epoch_number", config.EpochBlock)
	}
	return defaultDNR
}

func (d *DNR) Watch(ctx context.Context, db ethdb.Database) {
	d.syncLock.Lock()
	d.synced = false
	d.syncLock.Unlock()
	client, err := ethclient.DialContext(ctx, d.config.API)
	if err != nil {
		panic(err)
	}
	lastBlock := d.LastEpochBlock

	log.Warn("starting watch...", "existing_validators", d.Validators, "last_epoch", d.LastEpochBlock, "dnr_addr", d.config.DNR)

	for {
		startBlockNumber := new(big.Int).SetUint64(lastBlock + 1)
		lastBlockNumber := new(big.Int).Add(startBlockNumber, big.NewInt(2000))

		latestBlock, err := client.BlockByNumber(ctx, nil)
		if err != nil {
			panic(err)
		}

		// a 100 block delay to handle reorgs
		latestBlockNum := new(big.Int).Sub(latestBlock.Number(), big.NewInt(10))

		if latestBlockNum.Cmp(startBlockNumber) < 0 {
			// sleep for some time before retrying as no new block were created
			log.Warn("waiting for more new blocks....", "latest_block", latestBlockNum.Uint64(), "start_block", startBlockNumber.Uint64())
			time.Sleep(time.Minute)
			continue
		}

		if latestBlockNum.Cmp(lastBlockNumber) < 0 {
			lastBlockNumber = latestBlockNum
		}

		logs, err := client.FilterLogs(ctx, ethereum.FilterQuery{
			FromBlock: startBlockNumber,
			ToBlock:   lastBlockNumber,
			Addresses: []common.Address{d.config.DNR},
		})

		if err != nil {
			log.Warn("[x] error syncing", "err", err)
			panic(err)
		}

		for _, eventLog := range logs {
			switch eventLog.Topics[0].Hex() {
			case LogDarknodeRegistered:
				log.Warn("queuing pending darknode registration....", "darknode", common.BytesToAddress(eventLog.Topics[2].Bytes()))
				d.Validators[common.BytesToAddress(eventLog.Topics[2].Bytes())] = true
			case LogDarknodeDeregistered:
				if _, ok := d.Validators[common.BytesToAddress(eventLog.Topics[2].Bytes())]; ok {
					log.Warn("queuing pending darknode de-registration....", "darknode", common.BytesToAddress(eventLog.Topics[2].Bytes()))
					delete(d.Validators, common.BytesToAddress(eventLog.Topics[2].Bytes()))
				}
			case LogNewEpoch:
				log.Warn("storing epoch event....", "epoch", eventLog.BlockNumber)
				d.LastEpochBlock = eventLog.BlockNumber
				if err = d.store(db); err != nil {
					log.Warn("failed to store epoch in db", "epoch_number", eventLog.BlockNumber)
				}
			}
		}

		log.Warn("syncing eth events....", "latest_block", latestBlockNum.Uint64(), "synced_to", lastBlockNumber.Uint64())

		// if caught up set synced to true
		if latestBlockNum.Cmp(lastBlockNumber) == 0 {
			d.syncLock.Lock()
			d.synced = true
			d.syncLock.Unlock()
		}

		lastBlock = lastBlockNumber.Uint64()

		time.Sleep(time.Second * 30)
	}
}

func (d *DNR) store(db ethdb.Database) error {
	blob, err := json.Marshal(d)
	if err != nil {
		return err
	}
	if err = db.Put([]byte(fmt.Sprintf("dnr-%v", d.LastEpochBlock)), blob); err != nil {
		return err
	}
	return db.Put([]byte("dnr-latest"), blob)
}

func (d *DNR) WaitSynced() {
	for {
		d.syncLock.RLock()
		synced := d.synced
		d.syncLock.RUnlock()

		if synced {
			return
		}

		log.Info("still syncing...")
		time.Sleep(10 * time.Second)
	}
}

func GetLatestDNR(db ethdb.Database) (*DNR, error) {
	blob, err := db.Get([]byte(fmt.Sprintf("dnr-latest")))
	if err != nil {
		return nil, err
	}
	dnr := DNR{}
	if err = json.Unmarshal(blob, &dnr); err != nil {
		return nil, err
	}
	return &dnr, nil
}

func GetDNR(db ethdb.Database, epoch uint64) (*DNR, error) {
	blob, err := db.Get([]byte(fmt.Sprintf("dnr-%v", epoch)))
	if err != nil {
		return nil, err
	}
	dnr := DNR{}
	if err = json.Unmarshal(blob, &dnr); err != nil {
		return nil, err
	}
	return &dnr, nil
}
