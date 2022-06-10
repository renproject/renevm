pragma solidity 0.5.17;

contract DarknodeRegistryStateV1 {
    struct Epoch {
        uint256 epochhash;
        uint256 blocktime;
    }

    /// The current and previous epoch.
    Epoch public currentEpoch;
    Epoch public previousEpoch;

}

/// @notice DarknodeRegistry is responsible for the registration and
/// deregistration of Darknodes.
contract DarknodeRegistryLogicV1 is
    DarknodeRegistryStateV1
{
    /// @notice Emitted when a darknode is registered.
    /// @param _darknodeOperator The owner of the darknode.
    /// @param _darknodeID The ID of the darknode that was registered.
    /// @param _bond The amount of REN that was transferred as bond.
    event LogDarknodeRegistered(
        address indexed _darknodeOperator,
        address indexed _darknodeID,
        uint256 _bond
    );

    /// @notice Emitted when a darknode is deregistered.
    /// @param _darknodeOperator The owner of the darknode.
    /// @param _darknodeID The ID of the darknode that was deregistered.
    event LogDarknodeDeregistered(
        address indexed _darknodeOperator,
        address indexed _darknodeID
    );

    /// @notice Emitted when a new epoch has begun.
    event LogNewEpoch(uint256 indexed epochhash);

    constructor() public {
        uint256 epochhash = uint256(blockhash(block.number - 1));
        currentEpoch = Epoch({
            epochhash: epochhash,
            blocktime: block.timestamp
        });
        emit LogNewEpoch(epochhash);
    }

    function register(address _darknodeID, bytes calldata )
        external
    {
        // Emit an event.
        emit LogDarknodeRegistered(msg.sender, _darknodeID, 0);
    }

    function deregister(address _darknodeID)
        external
    {
        deregisterDarknode(_darknodeID);
    }

    function epoch() external {

        uint256 epochhash = uint256(blockhash(block.number - 1));

        // Update the epoch hash and timestamp
        previousEpoch = currentEpoch;
        currentEpoch = Epoch({
            epochhash: epochhash,
            blocktime: block.timestamp
        });

        // Emit an event
        emit LogNewEpoch(epochhash);
    }

    /// Private function called by `deregister` and `slash`
    function deregisterDarknode(address _darknodeID) private {
        // Emit an event
        emit LogDarknodeDeregistered(_darknodeID, _darknodeID);
    }
}