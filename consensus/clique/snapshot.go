// Copyright 2017 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package clique

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/params"
	lru "github.com/hashicorp/golang-lru"
	"sort"
)

// Vote represents a single vote that an authorized signer made to modify the
// list of authorizations.
type Vote struct {
	Signer    common.Address `json:"signer"`    // Authorized signer that cast this vote
	Block     uint64         `json:"block"`     // Block number the vote was cast in (expire old votes)
	Address   common.Address `json:"address"`   // Account being voted on to change its authorization
	Authorize bool           `json:"authorize"` // Whether to authorize or deauthorize the voted account
}

// Tally is a simple vote tally to keep the current score of votes. Votes that
// go against the proposal aren't counted since it's equivalent to not voting.
type Tally struct {
	Authorize bool `json:"authorize"` // Whether the vote is about authorizing or kicking someone
	Votes     int  `json:"votes"`     // Number of votes until now wanting to pass the proposal
}

// Snapshot is the state of the authorization voting at a given point in time.
type Snapshot struct {
	config   *params.CliqueConfig // Consensus engine parameters to fine tune behavior
	sigcache *lru.ARCCache        // Cache of recent block signatures to speed up ecrecover

	Number             uint64                    `json:"number"`                       // Block number where the snapshot was created
	PreviousSnapNumber *uint64                   `json:"previousSnapNumber,omitempty"` // previous snap block number
	PreviousSnapHash   *common.Hash              `json:"previousSnapHash,omitempty"`   // previous snap block hash
	EpochNumber        uint64                    `json:"epoch"`                        // dnr epoch when snapshot was created
	Hash               common.Hash               `json:"hash"`                         // Block hash where the snapshot was created
	Signers            map[common.Address]bool   `json:"signers"`                      // Set of authorized signers at this moment
	Recents            map[uint64]common.Address `json:"recents"`                      // Set of recent signers for spam protections
}

// signersAscending implements the sort interface to allow sorting a list of addresses
type signersAscending []common.Address

func (s signersAscending) Len() int           { return len(s) }
func (s signersAscending) Less(i, j int) bool { return bytes.Compare(s[i][:], s[j][:]) < 0 }
func (s signersAscending) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// newSnapshot creates a new snapshot with the specified startup parameters. This
// method does not initialize the set of recent signers, so only ever use if for
// the genesis block.
func newSnapshot(config *params.CliqueConfig, sigcache *lru.ARCCache, number, epoch uint64, previousNumber *uint64, hash common.Hash, previousHash *common.Hash, signers map[common.Address]bool) *Snapshot {
	snap := &Snapshot{
		config:             config,
		sigcache:           sigcache,
		Number:             number,
		Hash:               hash,
		EpochNumber:        epoch,
		Signers:            signers,
		Recents:            make(map[uint64]common.Address),
		PreviousSnapHash:   previousHash,
		PreviousSnapNumber: previousNumber,
	}
	return snap
}

// loadSnapshot loads an existing snapshot from the database.
func loadSnapshot(config *params.CliqueConfig, sigcache *lru.ARCCache, db ethdb.Database, number uint64) (*Snapshot, error) {
	blob, err := db.Get([]byte(fmt.Sprintf("clique-%v", number)))
	if err != nil {
		return nil, err
	}
	snap := new(Snapshot)
	if err := json.Unmarshal(blob, snap); err != nil {
		return nil, err
	}
	snap.config = config
	snap.sigcache = sigcache

	return snap, nil
}

// store inserts the snapshot into the database.
func (s *Snapshot) store(db ethdb.Database) error {
	blob, err := json.Marshal(s)
	if err != nil {
		return err
	}
	return db.Put([]byte(fmt.Sprintf("clique-%v", s.Number)), blob)
}

// copy creates a deep copy of the snapshot, though not the individual votes.
func (s *Snapshot) copy() *Snapshot {
	cpy := &Snapshot{
		config:      s.config,
		sigcache:    s.sigcache,
		Number:      s.Number,
		Hash:        s.Hash,
		Signers:     make(map[common.Address]bool),
		Recents:     make(map[uint64]common.Address),
		EpochNumber: s.EpochNumber,
	}
	for signer := range s.Signers {
		cpy.Signers[signer] = true
	}
	for block, signer := range s.Recents {
		cpy.Recents[block] = signer
	}

	return cpy
}

// signers retrieves the list of authorized signers in ascending order.
func (s *Snapshot) validEpoch(epochNum uint64, validatorBytes []byte, db ethdb.Database) (bool, map[common.Address]bool) {
	if s.EpochNumber >= epochNum {
		log.Warn("ignored epoch as current epoch >= proposed", "proposed", epochNum, "current", s.EpochNumber)
		return false, nil
	}
	dnr, err := GetDNR(db, epochNum)
	if err != nil {
		log.Warn("failed to get dnr snapshot for proposed epoch", "proposed", epochNum, "error", err.Error())
		return false, nil
	}
	proposedCount := len(validatorBytes) / common.AddressLength
	validators := []common.Address{}
	for i := 0; i < proposedCount; i++ {
		validators = append(validators, common.BytesToAddress(validatorBytes[i*common.AddressLength:(i+1)*common.AddressLength]))
	}

	if len(dnr.Validators) != len(validators) {
		log.Warn("validators for proposed epoch do not match stored data", "proposed", epochNum, "proposed_validators", validators, "stored_validators", dnr.Validators)
		return false, nil
	}
	for _, validator := range validators {
		if _, ok := dnr.Validators[validator]; !ok {
			log.Warn("validators for proposed epoch do not match stored data", "proposed", epochNum, "proposed_validators", validators, "stored_validators", dnr.Validators)
			return false, nil
		}
	}
	return true, dnr.Validators
}

// signers retrieves the list of authorized signers in ascending order.
func (s *Snapshot) signers() []common.Address {
	sigs := make([]common.Address, 0, len(s.Signers))
	for sig := range s.Signers {
		sigs = append(sigs, sig)
	}
	sort.Sort(signersAscending(sigs))
	return sigs
}

// inturn returns if a signer at a given block height is in-turn or not.
func (s *Snapshot) inturn(number uint64, signer common.Address) bool {
	signers, offset := s.signers(), 0
	for offset < len(signers) && signers[offset] != signer {
		offset++
	}
	return (number % uint64(len(signers))) == uint64(offset)
}

// signers retrieves the list of authorized signers in ascending order.
func (s *Snapshot) updateEpoch(header *types.Header, epoch uint64, signers map[common.Address]bool) {
	x, y := s.Number, s.Hash
	s.PreviousSnapNumber = &x
	s.PreviousSnapHash = &y
	s.Number = header.Number.Uint64()
	s.Hash = header.Hash()
	s.Signers = signers
	s.EpochNumber = epoch
}
